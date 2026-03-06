import os

def save_filenames_to_txt():
    # 1. PASTE YOUR FOLDER PATH BELOW THIS LINE
    pasted_path = r"truck_diagnostics_arma\cropped_parts\202603\parts_20260303_140812_506\wiper_debug_20260303_140812"
    
    # 2. AUTOMATIC PATH RESOLUTION (DO NOT EDIT BELOW)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the main project folder (parent of useful_tools)
    project_root = os.path.dirname(script_dir)
    project_name = os.path.basename(project_root)
    
    # Normalize slashes (convert all to forward slashes for comparison)
    pasted_path_normalized = pasted_path.replace('\\', '/')
    project_name_normalized = project_name.replace('\\', '/')
    
    # Check if the pasted path starts with the project name
    if pasted_path_normalized.startswith(project_name_normalized):
        # Remove the project name part to get the relative path
        relative_path = pasted_path_normalized[len(project_name_normalized):].lstrip('/')
        # Construct the real absolute path based on where the script actually is
        target_folder = os.path.join(project_root, relative_path)
    else:
        # If it doesn't start with the project name, use it as is (absolute path)
        target_folder = pasted_path

    # 3. Define the output text file path
    # Get the name of the target folder for the filename
    folder_name = os.path.basename(target_folder)
    output_filename = f"files_in_{folder_name}.txt"
    output_txt = os.path.join(script_dir, output_filename)

    # 4. Check if the folder exists
    if not os.path.exists(target_folder):
        print(f"Error: Folder not found at {target_folder}")
        print("Please check the path you pasted in the 'pasted_path' variable.")
        return

    # 5. Get all files and write them to the text file
    try:
        with open(output_txt, "w", encoding="utf-8") as f:
            # Write the folder name header first
            f.write(f"folder_name: {folder_name}\n")
            
            count = 0
            for item in os.listdir(target_folder):
                item_path = os.path.join(target_folder, item)
                # Only write if it is a file (ignore subfolders)
                if os.path.isfile(item_path):
                    f.write(item + "\n")
                    count += 1
        
        print(f"Success! Saved {count} filenames.")
        print(f"Output file created: {output_txt}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    save_filenames_to_txt()