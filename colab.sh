# u means updated files or new files will be copied only
rsync -aruvP --delete --exclude-from='.gitignore' --exclude='.git' ./* ~/gdrive-work/PyTorch_TrackNet/
