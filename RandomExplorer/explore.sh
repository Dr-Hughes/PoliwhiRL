cd ..
python ./main.py --use_config ./configs/explore.json
mv memory_data.db ./RandomExplorer/memory_data.db
cd RandomExplorer
python ./extract_imgs.py
echo "RandomExplorer Done"