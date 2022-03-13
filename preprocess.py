import os

from PIL import Image

classes = os.listdir('./data')

def crop_area(image, area: tuple, folder='', tag='', fn=''):
    w = area[0]
    h = area[1]

    im_size = image.size

    partitions = (im_size[0] / w, im_size[1] / h)

    top = 0
    left = 0

    h_crop_offset = 0
    v_crop_offset = 0
    
    file = image.filename.split('/')[-1]
    filename = file.split('.')[0] if not fn else fn

    for v in range(round(partitions[1])):
        for h in range(round(partitions[0])):
            right = left + area[0]
            lower = top + area[1]

            if partitions[0] - h_crop_offset < 0:
                print('outofbounds-h')

            if partitions[1] - v_crop_offset < 0:
                print('outofbounds-v')


            if partitions[0] - h_crop_offset < 1:
                right = im_size[0]
                left = im_size[0] - area[0]
            
            if partitions[1] - v_crop_offset < 1:
                lower = im_size[1]
                top = im_size[1] - area[1]

            crop = (left, top, right, lower)

            cropped = image.crop(crop)
            cropped.save(f'{folder}/{filename}-{h}{v}-cropped-{tag}.png')

            left += area[0]
            h_crop_offset += 1
            
        
        top += area[1]
        v_crop_offset += 1

        left = 0
        h_crop_offset = 0
            
def square_area(image, folder='', fn=''):
    im_size = image.size

    side = im_size[0] if im_size[0] < im_size[1] else im_size[1]

    crop_area(image, (side, side), folder=folder, tag=f'{side}crop', fn=fn) 

for c in classes:
    if not os.path.isdir(f'./training_data/{c}'):
        os.mkdir(f'./training_data/{c}')

file_number = 0
sizes = [
    (400, 400),
    (500, 500),
    (600, 600),
    (700, 700)
]
edit_number = 0

for c in classes:
    for f in os.listdir(f'./data/{c}'):
        file_number += 1
        print(f'file:{f} file_number:{file_number}')
        image = Image.open(f'./data/{c}/{f}')
        target_folder = f'./training_data/{c}'

        for s in sizes:
            edit_number += 1
            crop_area(image, s, folder=target_folder, tag=f'{s[0]}wh', fn=f'no{file_number}-ed{edit_number}')
        
        
        edit_number += 1
        square_area(image, folder=target_folder, fn=f'no{file_number}-ed{edit_number}')
        

        