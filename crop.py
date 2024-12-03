# import streamlit as st
# from streamlit_drawable_canvas import st_canvas
# from PIL import Image
# import numpy as np
# import cv2 as cv
# import os

# # Initialize models
# # simple_lama = SimpleLama()
# # model = FastSAM("FastSAM-s.pt")

# st.title("Crop Image")

# # Step 1: Upload an image
# uploaded_file = st.text_input('Give folder Path')

# if uploaded_file is not None:
#     # Load image and display it
#     img = Image.open(uploaded_file)
#     if img.size[0]>700:
#         height=img.size[1]
#         width=img.size[0]
#         new_width  = 700
#         new_height = int(new_width * height / width) 
#     img = img.resize((new_width, new_height), Image.LANCZOS)
#     img_np = np.array(img)
#     img_np2 = np.array(img)
#     st.write("Draw a Recatangle To remove object or choose an area")

#     # Set canvas for drawing bounding boxes
#     canvas_result = st_canvas(
#         fill_color="rgba(255, 0, 0, 0.3)",  # Transparent red fill for boxes
#         stroke_width=2,
#         background_image=img,
#         update_streamlit=True,
#         height=img.size[1],
#         width=img.size[0],
#         drawing_mode="rect",  # Enable rectangle mode
#         key="canvas",
#     )

#     # Get bounding box coordinates from the canvas
#     if canvas_result.json_data is not None:
#         for obj in canvas_result.json_data["objects"]:
#             if obj["type"] == "rect":
#                 # Get bounding box coordinates
#                 x = int(obj["left"])
#                 y = int(obj["top"])
#                 w = int(obj["width"])
#                 h = int(obj["height"])
#                 # st.write(f"Bounding box: x={x}, y={y}, width={w}, height={h}")
#                 bbox_drawn = True
#                 break
#         else:
#             bbox_drawn = False
#     else:
#         bbox_drawn = False

# import streamlit as st
# from streamlit_drawable_canvas import st_canvas
# from PIL import Image
# import numpy as np
# import os

# # Set up state variables for navigation
# if "current_index" not in st.session_state:
#     st.session_state.current_index = 0

# # App title
# st.title("Image Cropping Tool")

# # Step 1: Provide folder path
# folder_path = st.text_input("Enter folder path containing images:")

# if folder_path and os.path.isdir(folder_path):
#     # Get list of images in the folder
#     image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
#     image_files.sort()  # Optional: Sort files for consistent order

#     if not image_files:
#         st.warning("No image files found in the specified folder.")
#     else:
#         # Navigation buttons
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col1:
#             if st.button("Previous"):
#                 st.session_state.current_index = max(0, st.session_state.current_index - 1)
#         with col3:
#             if st.button("Next"):
#                 st.session_state.current_index = min(len(image_files) - 1, st.session_state.current_index + 1)

#         # Load the current image
#         current_image_path = os.path.join(folder_path, image_files[st.session_state.current_index])
#         img = Image.open(current_image_path)
        
#         # Resize image if width > 700px
#         if img.size[0] > 700:
#             width, height = img.size
#             new_width = 700
#             new_height = int(new_width * height / width)
#             img = img.resize((new_width, new_height), Image.LANCZOS)

#         # Convert image to numpy array
#         img_np = np.array(img)

#         # Display the current image
#         # st.image(img, caption=f"Viewing: {image_files[st.session_state.current_index]}", use_column_width=True)
#         st.write("Draw a rectangle to crop a region from the image.")

#         # Set up canvas for drawing
#         canvas_result = st_canvas(
#             fill_color="rgba(255, 0, 0, 0.3)",  # Transparent red fill
#             stroke_width=2,
#             background_image=img,
#             update_streamlit=True,
#             height=img.size[1],
#             width=img.size[0],
#             drawing_mode="rect",
#             key=f"canvas_{st.session_state.current_index}",  # Unique key for each image
#         )

#         # Process bounding box and save cropped region
#         if canvas_result.json_data is not None:
#             for obj in canvas_result.json_data["objects"]:
#                 if obj["type"] == "rect":
#                     # Get bounding box coordinates
#                     x = int(obj["left"])
#                     y = int(obj["top"])
#                     w = int(obj["width"])
#                     h = int(obj["height"])

#                     # Crop the image
#                     cropped_img = img_np[y:y + h, x:x + w]

#                     # Create "crops" folder if it doesn't exist
#                     crops_folder = os.path.join(folder_path, "crops")
#                     os.makedirs(crops_folder, exist_ok=True)

#                     # Save the cropped region
#                     cropped_img_pil = Image.fromarray(cropped_img)
#                     save_path = os.path.join(crops_folder, f"crop_{image_files[st.session_state.current_index]}")
#                     cropped_img_pil.save(save_path)

#                     st.success(f"Cropped region saved to: {save_path}")
#                     break



import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import os

# Set up state variables for navigation
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "canvas_scale" not in st.session_state:
    st.session_state.canvas_scale = 1.0  # Default scale

# App title
st.title("Image Cropping Tool with Zoom")

# Step 1: Provide folder path
folder_path = st.text_input("Enter folder path containing images:")

if folder_path and os.path.isdir(folder_path):
    # Get list of images in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    image_files.sort()  # Optional: Sort files for consistent order

    if not image_files:
        st.warning("No image files found in the specified folder.")
    else:
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("Previous"):
                st.session_state.current_index = max(0, st.session_state.current_index - 1)
        with col3:
            if st.button("Next"):
                st.session_state.current_index = min(len(image_files) - 1, st.session_state.current_index + 1)

        # Load the current image
        current_image_path = os.path.join(folder_path, image_files[st.session_state.current_index])
        img = Image.open(current_image_path)
        
        # Resize image if width > 700px
        # if img.size[0] > 700:
        #     width, height = img.size
        #     new_width = 700
        #     new_height = int(new_width * height / width)
        #     img = img.resize((new_width, new_height), Image.LANCZOS)

        # Convert image to numpy array
        img_np = np.array(img)

        # Zoom Controls
        st.sidebar.title("Zoom Controls")
        zoom_factor = st.sidebar.slider("Zoom Level", min_value=0.5, max_value=3.0, value=st.session_state.canvas_scale, step=0.1)
        st.session_state.canvas_scale = zoom_factor

        # Scale the image for the canvas
        canvas_width = int(img.size[0] * zoom_factor)
        canvas_height = int(img.size[1] * zoom_factor)
        resized_img = img.resize((canvas_width, canvas_height), Image.LANCZOS)

        # Display the current image
        # st.image(img, caption=f"Viewing: {image_files[st.session_state.current_index]}", use_column_width=True)
        st.write("Draw a rectangle to crop a region from the image.")

        # Set up canvas for drawing
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",  # Transparent red fill
            stroke_width=2,
            background_image=resized_img,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            key=f"canvas_{st.session_state.current_index}",  # Unique key for each image
        )

        # Process bounding box and save cropped region
        if canvas_result.json_data is not None:
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "rect":
                    # Get bounding box coordinates (adjust for zoom)
                    x = int(obj["left"] / zoom_factor)
                    y = int(obj["top"] / zoom_factor)
                    w = int(obj["width"] / zoom_factor)
                    h = int(obj["height"] / zoom_factor)

                    # Crop the image
                    cropped_img = img_np[y:y + h, x:x + w]

                    # Create "crops" folder if it doesn't exist
                    crops_folder = os.path.join(folder_path, "crops")
                    os.makedirs(crops_folder, exist_ok=True)

                    # Save the cropped region
                    cropped_img_pil = Image.fromarray(cropped_img)
                    save_path = os.path.join(crops_folder, f"crop_{image_files[st.session_state.current_index]}")
                    cropped_img_pil.save(save_path)

                    st.success(f"Cropped region saved to: {save_path}")
                    break

