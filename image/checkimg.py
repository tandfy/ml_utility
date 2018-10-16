def __read_lst(dat):
    """
    lst形式のデータ(文字列)の内容を読み込む
    """
    dat_list = dat.split('\t')
    index = int(dat_list[0])

    header_size = int(dat_list[1])
    assert header_size == 2, 'header_sizeは２を想定:'+str(header_size)

    label_width = int(dat_list[2])
    assert label_width == 5, 'label_widthは5を想定: '+str(label_width)

    label_data = dat_list[3:-1]
    assert (len(label_data) % label_width) == 0 , 'label_dataの長さはlabel_widthの倍数のはず : '

    file_path = dat_list[-1]

    return (index, header_size, label_width, label_data, file_path)

def create_bb_img(input_lst_path, input_img_root_path, output_img_path, class_list=[]):
    """
    画像データにlstファイルに基づくバウンディングボックスを加工した画像をつうる
    """

    import random
    import copy
    import os
    from os import path
    import shutil
    from PIL import Image
    import numpy as np
    import imgaug as ia
    from tqdm import tqdm_notebook as tqdm

    # 出力先をリセット
    if path.isdir(output_img_path):
        shutil.rmtree(output_img_path)

    os.makedirs(output_img_path)

    with open(input_lst_path) as lst_f:
        for line in tqdm(lst_f.readlines()):
            line = line.strip()
            if not line: continue

            #lst形式のデータを読み取って、変数に入れる
            origin_img_index, header_size, label_width, label_data, img_path = __read_lst(line)
            img_path = path.join(input_img_root_path, img_path)


            # 画像を読み込む
            origin_img = Image.open(img_path).convert('RGB')
            img_height = origin_img.height
            img_width = origin_img.width
            max_edge = max(img_height, img_width)

            # 画像を変換する
            target_img = np.array(origin_img)

            # バウンディングボックスを生成
            bbs = []
            for bb_index in range(len(label_data)//label_width):
                                           
                bb = ia.BoundingBox(
                    x1 = float(label_data[bb_index * label_width + 1]) * img_width,
                    y1 = float(label_data[bb_index * label_width + 2]) * img_height,
                    x2 = float(label_data[bb_index * label_width + 3]) * img_width,
                    y2 = float(label_data[bb_index * label_width + 4]) * img_height
                )
                class_val = int(label_data[bb_index * label_width])
                assert 0 <= class_val and class_val < len(class_list), 'classの値が不正です。 : '+str(class_val)
                class_name = class_list[class_val] if class_list[class_val] else str(class_val)
                target_img = ia.draw_text(target_img, bb.y1, bb.x1, class_name)
                
                bbs.append(bb)
                

            bbs_on_img = ia.BoundingBoxesOnImage(bbs, shape = target_img.shape)

            after_bb_img = bbs_on_img.draw_on_image(target_img)
            output_img_name = path.basename(img_path)
            Image.fromarray(after_bb_img).save(path.join(output_img_path, output_img_name))
