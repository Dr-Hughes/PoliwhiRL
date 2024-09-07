#!/bin/bash

# Directory to encrypt
DIR_TO_ENCRYPT="../emu_files"

# Generate a random passphrase
PASSPHRASE=$(openssl rand -base64 32)

# Ensure the directory exists
if [ ! -d "$DIR_TO_ENCRYPT" ]; then
    echo "Directory $DIR_TO_ENCRYPT does not exist."
    exit 1
fi

# Create a directory for encrypted files
ENCRYPTED_DIR="${DIR_TO_ENCRYPT}_encrypted"
mkdir -p "$ENCRYPTED_DIR"

# Encrypt each file in the directory
for file in "$DIR_TO_ENCRYPT"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        gpg --symmetric --cipher-algo AES256 --batch --passphrase "$PASSPHRASE" \
            --output "$ENCRYPTED_DIR/$filename.gpg" "$file"
        echo "Encrypted $filename"
    fi
done

# Save the passphrase (you'll add this to GitHub Secrets later)
echo "Passphrase: $PASSPHRASE"
echo "Save this passphrase securely and add it to your GitHub Secrets."

echo "Encryption complete. Encrypted files are in $ENCRYPTED_DIR"