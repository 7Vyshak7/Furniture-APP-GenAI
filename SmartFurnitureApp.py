import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2 as cv
from simple_lama_inpainting import SimpleLama
from ultralytics import FastSAM
import os

# Initialize models
simple_lama = SimpleLama()
model = FastSAM("FastSAM-s.pt")

st.title("Smart Furnish")

# Step 1: Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image and display it
    img = Image.open(uploaded_file)
    if img.size[0]>700:
        height=img.size[1]
        width=img.size[0]
        new_width  = 700
        new_height = int(new_width * height / width) 
    img = img.resize((new_width, new_height), Image.LANCZOS)
    img_np = np.array(img)
    img_np2 = np.array(img)
    st.write("Draw a Recatangle To remove object or choose an area")

    # Set canvas for drawing bounding boxes
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # Transparent red fill for boxes
        stroke_width=2,
        background_image=img,
        update_streamlit=True,
        height=img.size[1],
        width=img.size[0],
        drawing_mode="rect",  # Enable rectangle mode
        key="canvas",
    )

    # Get bounding box coordinates from the canvas
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "rect":
                # Get bounding box coordinates
                x = int(obj["left"])
                y = int(obj["top"])
                w = int(obj["width"])
                h = int(obj["height"])
                # st.write(f"Bounding box: x={x}, y={y}, width={w}, height={h}")
                bbox_drawn = True
                break
        else:
            bbox_drawn = False
    else:
        bbox_drawn = False

    # Step 3: Detect objects in the bounding box
    if bbox_drawn and st.button("Start Deteting Fetaures"):
        with st.spinner("Please Wait"):
            bboxes = [[x, y, x + w, y + h]]
            st.session_state['x']=x
            st.session_state['y']=y
            results = model(img_np, bboxes=bboxes,retina_masks=True, conf=0.2, iou=0.5)

            # Create a mask from FastSAM results
            mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
            for result in results:
                segments = result.masks.xy
                boxes = result.boxes.xyxy
                for seg2, box in zip(segments, boxes):
                    inside=True
                    silp = np.array(seg2)
                    x1, y1, x2, y2 = map(int, box)
                    for xs,ys in seg2:
                        # print(xs,ys)
                        if (x > xs or xs > x+w) or (y > ys or ys > y+h):
                            inside=False
                    if inside:
                        mask[y1:y2, x1:x2] = 255
                        st.session_state['inside'] = inside
                        cv.fillPoly(img_np, pts=[silp.astype('int32')], color=(255, 0, 0))
                        
                    else:
                        mask[y:y+h, x:x+w] = 255
                    continue

            img3=cv.addWeighted(img_np, 0.5, img_np2, 0.5, 0.0)
            # Store mask in session state for later use
            st.session_state['mask'] = mask
            
            
            # Display the image with bounding box overlay
            st.image(img3, caption="Area to be Removed", use_column_width=False)

        if 'inside' not in st.session_state:
            st.write('No Object to be removed from the area')
        # if 'inside' not in st.session_state:
        #     print('enetered in inpainted code')
            st.session_state['inpainted_image'] = img_np2
        

    # Step 4: Apply inpainting
    if 'inside' in st.session_state and 'mask' in st.session_state and st.button("Start Removing") :
        with st.spinner("Please Wait"):
            mask = st.session_state['mask']
            img_pil = Image.fromarray(cv.cvtColor(img_np, cv.COLOR_BGR2RGB))
            inpainted_result = simple_lama(img_pil, mask)

            # Display the inpainting result
            inpainted_np = np.array(inpainted_result)
            result_rgb = cv.cvtColor(np.array(inpainted_np), cv.COLOR_BGR2RGB)
            st.session_state['inpainted_image'] = inpainted_np
            st.image(result_rgb, caption="Removal Result", use_column_width=True)
            st.success("Removing completed.")
    

    # Step 5: Choose and Place Furniture Option
    if 'inpainted_image' in st.session_state:
        st.sidebar.header("Choose a Furniture Option")
        option = st.sidebar.selectbox("Select Furniture Type", ["Chair", "Table", "Other Furniture"])

        # Sample furniture images (replace with your actual image paths)
        furniture_images = {
            "Chair": ["./Items/c1.jpg", "./Items/c2.jpg", "./Items/c3.jpg"],
            "Table": ["./Items/T1.jpg", "./Items/T2.jpg", "./Items/T3.jpg"],
            "Other Furniture": ["./Items/F1.jpg", "./Items/F2.jpg", "./Items/F3.jpg"],
        }

        selected_images = furniture_images.get(option, [])
        chosen_image = None

        if selected_images:
            # Display sample images in the sidebar
            for i, image_path in enumerate(selected_images):
                st.sidebar.image(image_path, caption=f"{option} {i + 1}", width=100)
                if st.sidebar.button(f"Select {option} {i + 1}", key=image_path):
                    chosen_image = image_path
                    st.session_state['chosen_furniture'] = chosen_image
    if 'chosen_furniture' in st.session_state:
            chosen_furniture_image = Image.open(st.session_state['chosen_furniture'])
            chosen_furniture_np = np.array(chosen_furniture_image)

            # Step 5.1: Run FastSAM on the chosen furniture image
            with st.spinner("Adding Furniture to your Area"):
                bboxes = [[0, 0, chosen_furniture_np.shape[1], chosen_furniture_np.shape[0]]]
                results = model(chosen_furniture_np, bboxes=bboxes)

                # Create a blank mask for the chosen furniture segmentation
                furniture_mask = np.zeros(chosen_furniture_np.shape[:2], dtype=np.uint8)

                # Extract only the `seg2` polygon mask
                for result in results:
                    segments = result.masks.xy
                    for seg2 in segments:
                        silp = np.array(seg2)
                        # Fill the mask with the segmented part
                        cv.fillPoly(furniture_mask, pts=[silp.astype('int32')], color=255)

                # Apply mask to extract only the segmented part of the furniture
                segmented_furniture = cv.bitwise_and(chosen_furniture_np, chosen_furniture_np, mask=furniture_mask)

                # Step 5.2: Resize the segmented furniture to fit the bounding box area
                resized_furniture = cv.resize(segmented_furniture, (w, h))
                if 'inside' in st.session_state:
                    resized_furniture=cv.cvtColor(resized_furniture,cv.COLOR_BGR2RGB)

                # Prepare the inpainted image for placement
                result_img = st.session_state['inpainted_image'].copy()


                # Only place the segmented part (non-zero mask values) into the inpainted area
                mask_resized = cv.resize(furniture_mask, (w, h))
                st.session_state['resized_mask']=mask_resized
                for i in range(3):  # Apply the mask to each color channel
                    result_img[y:y+h, x:x+w, i] = np.where(mask_resized == 255, resized_furniture[:, :, i], result_img[y:y+h, x:x+w, i])

                # Display the final result
                st.session_state['resized_furniture'] = resized_furniture
                if 'inside' in st.session_state:
                    result_img_rgb = cv.cvtColor(np.array(result_img), cv.COLOR_BGR2RGB)
                else:
                    result_img_rgb = np.array(result_img)
                st.image(result_img_rgb, caption="Process Completed", use_column_width=True)



    # Step 6: Drag and Place the Furniture
    if 'resized_furniture' in st.session_state:
        st.write("Step 6: Drag and place the furniture on the canvas (Scroll to resize)")

        # Initialize placement and resizing variables
        if 'furniture_placed' not in st.session_state:
            st.session_state['furniture_placed'] = False
            st.session_state['mouse_down'] = False
            st.session_state['offset_x'] = 0
            st.session_state['offset_y'] = 0
            st.session_state['scale_factor'] = 1.0

        # Load images
        inpainted_image = st.session_state['inpainted_image'].copy()
        resized_furniture = st.session_state['resized_furniture']
        original_furniture = st.session_state['resized_furniture'].copy()
        mask = st.session_state['resized_mask']

        original_mask = st.session_state['resized_mask'].copy()
        x, y = st.session_state['x'], st.session_state['y']  # Initial top-left coordinates for the furniture

        def drag_and_resize(event, mx, my, flags, param):
            global x, y, resized_furniture, mask

            # Dragging logic
            if event == cv.EVENT_LBUTTONDOWN:
                # Check if the mouse is within the furniture area
                if (x <= mx <= x + resized_furniture.shape[1] and y <= my <= y + resized_furniture.shape[0]):
                    st.session_state['mouse_down'] = True
                    st.session_state['offset_x'] = mx - x
                    st.session_state['offset_y'] = my - y

            elif event == cv.EVENT_MOUSEMOVE and st.session_state['mouse_down']:
                # Update position while dragging
                x = mx - st.session_state['offset_x']
                y = my - st.session_state['offset_y']

            elif event == cv.EVENT_LBUTTONUP:
                st.session_state['mouse_down'] = False

            # Resizing 
            elif event == cv.EVENT_MOUSEWHEEL:
                # Determine scroll direction and update scale factor
                if flags > 0:  # Scroll up
                    st.session_state['scale_factor'] += 0.1
                else:  # Scroll down
                    st.session_state['scale_factor'] = max(0.1, st.session_state['scale_factor'] - 0.1)

                # Resize the furniture and mask
                new_width = int(original_furniture.shape[1] * st.session_state['scale_factor'])
                new_height = int(original_furniture.shape[0] * st.session_state['scale_factor'])

                # Resize furniture and mask based on scale factor
                resized_furniture = cv.resize(original_furniture, (new_width, new_height), interpolation=cv.INTER_AREA)
                mask = cv.resize(original_mask, (new_width, new_height), interpolation=cv.INTER_AREA)

        # Create a window for interactive placement and resizing
        cv.namedWindow('Place and Resize Furniture')
        cv.setMouseCallback('Place and Resize Furniture', drag_and_resize)

        while True:
            # Create a copy of the inpainted image for display
            display_image = inpainted_image.copy()
            display_image2=inpainted_image.copy()

            # Overlay the resized furniture image using the mask
            for i in range(3):
                y1, y2 = max(0, y), min(display_image.shape[0], y + resized_furniture.shape[0])
                x1, x2 = max(0, x), min(display_image.shape[1], x + resized_furniture.shape[1])
                fy1, fy2 = 0, y2 - y1
                fx1, fx2 = 0, x2 - x1

                display_image[y1:y2, x1:x2, i] = np.where(
                    mask[fy1:fy2, fx1:fx2] == 255,
                    resized_furniture[fy1:fy2, fx1:fx2, i],
                    display_image[y1:y2, x1:x2, i]
                )
                display_image2[y1:y2, x1:x2, i] = np.where(
                    mask[fy1:fy2, fx1:fx2] == 255,
                    resized_furniture[fy1:fy2, fx1:fx2, i],
                    display_image2[y1:y2, x1:x2, i]
                )

            # Display the image
            cv.putText(display_image, "Press 's' to save the image after resizing/moving object", 
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
            if 'inside' not in st.session_state:
                display_image=cv.cvtColor(display_image, cv.COLOR_BGR2RGB)
                display_image2=cv.cvtColor(display_image2, cv.COLOR_BGR2RGB)
            cv.imshow('Place and Resize Furniture', display_image)

            # Wait for user to press 'q' to confirm placement
            if cv.waitKey(1) & 0xFF == ord('s'):
                st.session_state['final_image'] = display_image2
                cv.destroyAllWindows()
                break

        # Display the final image in Streamlit
        if st.session_state['final_image'].any():
            final_image_rgb = cv.cvtColor(st.session_state['final_image'], cv.COLOR_BGR2RGB)
            # if 'inside' not in 
            st.image(final_image_rgb, caption="Final Image with Placed and Resized Furniture", use_column_width=True)
            st.success("Furniture placement and resizing completed.")