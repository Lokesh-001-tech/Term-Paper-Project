import os


raw_folder = "data/raw-890"
reference_folder = "data/reference-890"

raw_images = os.listdir(raw_folder)
reference_images = os.listdir(reference_folder)

raw_images = set(raw_images)
reference_images = set(reference_images)

print("Raw images:", len(raw_images))
print("Reference images:", len(reference_images))

missing_references = raw_images - reference_images
missing_raw = reference_images - raw_images

print("\nRaw images without references:")
print(missing_references)

print("\nReferences without raw images:")
print(missing_raw)

if not missing_references and not missing_raw:
    print("\nAll filenames are correctly matched!")
else:
    print("\nSome images are not matched.")