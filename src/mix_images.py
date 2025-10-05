import argparse
import mimetypes
import os
import time
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash-image-preview"
DEFAULT_IMAGE_FOLDER = "images"

# Define the number of iterations for a single image
SINGLE_IMAGE_ITERATIONS = 3


def stage_images(
    image_paths: list[str],
    prompt: str,
    output_dir: str,
):
    """
    Does virtual staging for property images using the Google Generative AI model.

    Args:
        image_paths: A list of two paths to input images.
        prompt: The prompt for remixing the images.
        output_dir: Directory to save the remixed images.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    contents = _load_image_parts(image_paths)
    contents.append(genai.types.Part.from_text(text=prompt))

    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
    )

    # Note: When called from the new loop in main(), image_paths will only contain one path.
    print(f"Staging {len(image_paths)} image(s): {image_paths[0] if image_paths else 'None'}")

    stream = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=generate_content_config,
    )

    _process_api_stream_response(stream, output_dir)


def _load_image_parts(image_paths: list[str]) -> list[types.Part]:
    """Loads image files and converts them into GenAI Part objects."""
    parts = []
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image_data = f.read()
        mime_type = _get_mime_type(image_path)
        parts.append(
            types.Part(inline_data=types.Blob(data=image_data, mime_type=mime_type))
        )
    return parts


def _process_api_stream_response(stream, output_dir: str):
    """Processes the streaming response from the GenAI API, saving images and printing text."""
    # We remove the timestamp from the filename to better differentiate the iteration outputs
    # and ensure the file names are unique enough.
    file_index = 0
    for chunk in stream:
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue

        for part in chunk.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                # Use a high-resolution timestamp/random number for uniqueness
                unique_id = int(time.time() * 1000)
                file_extension = mimetypes.guess_extension(part.inline_data.mime_type)
                
                # The name format will be handled by the calling function (main)
                # For now, we'll keep the basic unique naming scheme
                file_name = os.path.join(
                    output_dir,
                    f"remixed_image_{unique_id}_{file_index}{file_extension}",
                )
                _save_binary_file(file_name, part.inline_data.data)
                file_index += 1
            elif part.text:
                print(part.text)


def _save_binary_file(file_name: str, data: bytes):
    """Saves binary data to a specified file."""
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")


def _get_mime_type(file_path: str) -> str:
    """Guesses the MIME type of a file based on its extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {file_path}")
    return mime_type


def _get_images_from_folder(folder_path: str) -> list[str]:
    """
    Collects valid image file paths from a specified folder,
    sorted alphabetically.
    """
    if not os.path.isdir(folder_path):
        return []

    image_paths = []
    # List files and sort them to ensure a consistent order
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                # Check if it's an image file
                mime_type = _get_mime_type(file_path)
                if mime_type.startswith('image/'):
                    image_paths.append(file_path)
            except ValueError:
                # Ignore files for which MIME type can't be determined
                continue
    return image_paths


def main():
    parser = argparse.ArgumentParser(
        description="Stages images using Google Generative AI."
    )
    parser.add_argument(
        "-i",
        "--image",
        action="append",
        # Set required=False to make the argument optional
        required=False,
        help=f"Paths to input images (1-5 images). If not provided, all images in the '{DEFAULT_IMAGE_FOLDER}' folder will be used.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Optional prompt for staging the images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save the staged images.",
    )

    

    args = parser.parse_args()

    # --- Logic to handle image paths ---
    all_image_paths = args.image
    if all_image_paths is None:
        # If no -i/--image arguments are passed, use images from the default folder
        print(f"No image paths provided. Searching for images in the '{DEFAULT_IMAGE_FOLDER}' folder...")
        all_image_paths = _get_images_from_folder(DEFAULT_IMAGE_FOLDER)

    # Check for valid number of images
    num_images = len(all_image_paths)
    if num_images == 0:
        parser.error(
            f"No images found. Please provide images using the -i flag or place 1 to 5 images in the '{DEFAULT_IMAGE_FOLDER}' folder."
        )
    elif num_images > 5:
        print("⚠️ Warning: You have more than 5 images. The script will process the first 5 to adhere to the recommended limit.")
        all_image_paths = all_image_paths[:5]

    # Determine the prompt
    final_prompt = args.prompt
    if final_prompt is None:
        final_prompt = "Add some premium decor to this image. It should be neutral and minimalistic. Maintain an appropriate sense of scale and sense of space within the room. Do not add or alter permanent features (cabinets, windows, curtains, light fixtures, walls, flooring, power outlets). If the type of the object is unclear, leave it unchanged."


    # Ensure output directory exists
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Start Image Staging Loop ---
    print(f"\n--- Starting Individual Image Staging for {len(all_image_paths)} image(s) ---\n")
    
    # Determine the number of times to run for *each* image
    # If there is only one image, run it three times (SINGLE_IMAGE_ITERATIONS).
    # Otherwise (multiple images), run each image only once.
    runs_per_image = SINGLE_IMAGE_ITERATIONS if len(all_image_paths) == 1 else 1

    for i, image_path in enumerate(all_image_paths):
        # The inner loop handles the multiple run requirement for a single image
        for j in range(runs_per_image):
            run_info = ""
            if runs_per_image > 1:
                run_info = f" (Iteration {j + 1} of {runs_per_image})"
            
            print(f"\nProcessing Image {i + 1} of {len(all_image_paths)}: {os.path.basename(image_path)}{run_info}")

            # Call stage_images with a list containing only the current image path
            stage_images(
                image_paths=[image_path],
                prompt=final_prompt,
                output_dir=output_dir,
            )
            
        print(f"Finished all runs for {image_path}.")

    print("\n--- All staging tasks complete! ---")


if __name__ == "__main__":
    main()