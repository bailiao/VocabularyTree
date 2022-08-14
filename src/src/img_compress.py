import os
from PIL import Image

# conditional compress imgs in one folder into another folder
def compressImgs(size_cut, compress_quality, resized_ratio, origin_path, compressed_path):
    if not os.path.exists(compressed_path):
        os.mkdir(compressed_path)

    file_list = os.listdir(origin_path)
    for file in file_list:
        origin_img_path = origin_path + '/' + file 
        compress_img_path = compressed_path + '/' + file
        img = Image.open(origin_img_path)
        if os.path.getsize(origin_img_path) > size_cut*1024*1024:
            compress_img = img.resize(
                ( (int)(img.size[0] // resized_ratio), (int)(img.size[1] // resized_ratio) ),
                Image.ANTIALIAS)
            compress_img.save(compress_img_path, quality = compress_quality)
        else:
            img.save(compress_img_path, quality = compress_quality)
    



