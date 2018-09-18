def __create_lst(file_path, index, label_data):
    """
    lst形式のデータを作成
    """

    header_size = 2
    label_width = 5
    return '\t'.join([
        str(index),
        str(header_size),
        str(label_width),
        '\t'.join( label_data),
        file_path])

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




def merge_annotated_img(lst_path_list, img_root_path_list, merged_root_path):
    """
    複数の教師画像データセット(lstと画像)を一つにまとめる
    """

    import os
    from os import path
    import shutil
    from PIL import Image
    from tqdm import tqdm_notebook as tqdm

    assert len(lst_path_list) == len(img_root_path_list), "lst_path_listとimg_root_path_listの長さは同じのはず"


    #マージしたlstファイルと画像ファイルの出力先
    output_lst_path = path.join(merged_root_path, "lst.lst")
    output_img_root_path = path.join(merged_root_path, "img")

    # 出力先をリセット
    if path.isdir(merged_root_path):
        shutil.rmtree(merged_root_path)

    os.makedirs(merged_root_path)
    os.makedirs(output_img_root_path)

    # マージ処理開始
    merged_lst = []
    for lst_path, img_root_path in tqdm(zip(lst_path_list, img_root_path_list)):
        with open(lst_path) as lst_f:
            for line in tqdm(lst_f.readlines()):
                line = line.strip()
                if not line: continue

                #lst形式のデータを読み取って、変数に入れる
                index, header_size, label_width, label_data, img_path = __read_lst(line)
                img_path = path.join(img_root_path, img_path)

                merged_index = len(merged_lst) + 1


                # 画像ファイル名をcountに書き換える
                after_img_name =  str(merged_index) + path.splitext(img_path)[1]
                after_img_path = path.join(output_img_root_path, after_img_name)

                #マージ後ファイル出力先へ画像をコピー
                shutil.copy(img_path, after_img_path)


                #lst形式のテキストを作成
                lst_dat = __create_lst(after_img_name, merged_index, label_data)
                merged_lst.append(lst_dat)


    # 作成したデータを要素ごとに改行して書き出す
    with open(output_lst_path, 'w') as out_f:
        out_f.write('\n'.join(merged_lst))


def resize_img_square(pil_img, edge_size, background_color=(0, 0, 0)):
    """
    一辺がedge_sizeの正方形にリサイズする
    """

    from PIL import Image
    longer_edge = max(pil_img.size)
    shorter_edge = min(pil_img.size)

    if longer_edge == shorter_edge:
        return pil_img.resize((edge_size, edge_size))
    else:
        square_img = Image.new(pil_img.mode, (longer_edge, longer_edge), background_color)
        # 正方形の背景画像の左上に画像を配置
        square_img.paste(pil_img, (0,0))
        return square_img.resize((edge_size, edge_size))

def process_image_for_classification(validate_ratio, img_augmentors, img_edge_size, input_lst_path, input_img_root_path, output_root_path, except_class_list=[], test_ratio=0, validate_augment=True, test_augment=False):
    """
    画像分類に使用する画像とlstデータの加工

    画像増幅と、検証/学習用へのデータ分けを行う。
    戻り値でtrainとvalのそれぞれのデータ数を返す。

    """

    import random
    import os
    from os import path
    import shutil
    from PIL import Image
    from tqdm import tqdm_notebook as tqdm
    import numpy as np

    assert validate_ratio + test_ratio < 1.0, "validate_ratio +test_ratioは1以下のはず {} , {}".format(validate_ratio, test_ratio)

    #マージした画像ファイルの出力先
    train_output_img_root_path = path.join(output_root_path, "train")

    val_output_img_root_path = path.join(output_root_path, "val")
    test_output_img_root_path = path.join(output_root_path, "test")


    # 出力先をリセット
    if path.isdir(output_root_path):
        shutil.rmtree(output_root_path)

    os.makedirs(output_root_path)
    os.makedirs(train_output_img_root_path)
    os.makedirs(val_output_img_root_path)
    os.makedirs(test_output_img_root_path)

    val_cnt_dic = {}
    train_cnt_dic = {}
    test_cnt_dic = {}
    with open(input_lst_path) as lst_f:
        for line in tqdm(lst_f.readlines()):
            line = line.strip()
            if not line: continue

            #lst形式のデータを読み取って、変数に入れる
            origin_img_index, header_size, label_width, label_data, img_path = __read_lst(line)
            img_path = path.join(input_img_root_path, img_path)

            # 画像を読み込む
            origin_img = Image.open(img_path)
            origin_img = origin_img.convert('RGB')

            # バウンディングボックスから該当オブジェクトを切り取る
            img_width = origin_img.width
            img_height = origin_img.height
            bbs = []
            for bb_index in range(len(label_data)//label_width):
                # データ取得
                label = label_data[bb_index * label_width]
                left = float(label_data[bb_index * label_width + 1]) * img_width
                upper = float(label_data[bb_index * label_width + 2]) * img_height
                right = float(label_data[bb_index * label_width + 3]) * img_width
                lower = float(label_data[bb_index * label_width + 4]) * img_height

                if label in except_class_list:
                    continue

                # 画像を切り取る
                target_img_ar = np.array(resize_img_square(origin_img.crop((left, upper, right, lower)), img_edge_size))


                # 画像の増幅数
                aug_num = len(img_augmentors)

                # カウンティング用辞書の初期化
                if label not in val_cnt_dic:
                    val_cnt_dic[label] = 0
                    test_cnt_dic[label] = 0
                    train_cnt_dic[label] = 0

                # 指定した確率で検証用データとして割り当てる
                rand_num = random.random()
                if rand_num < validate_ratio: # 検証用
                    output_img_root_path = val_output_img_root_path
                    cnt_dic = val_cnt_dic
                    is_augment = validate_augment

                elif rand_num < test_ratio + validate_ratio: # テスト用
                    output_img_root_path = test_output_img_root_path
                    cnt_dic = test_cnt_dic
                    is_augment = test_augment

                else: # 学習用
                    output_img_root_path = train_output_img_root_path
                    cnt_dic = train_cnt_dic
                    is_augment = True # 学習用は確定で増幅する


                # 保存先の確認
                target_img_dir = path.join(output_img_root_path, label)
                if not path.isdir(target_img_dir):
                    os.makedirs(target_img_dir)

                #画像増幅が不要な場合はそのまま保存
                if not is_augment or len(img_augmentors) == 0:
                    after_img_name = "{0:05d}_{1:02d}_{2:02d}{3}".format(origin_img_index, bb_index+1, 1, path.splitext(img_path)[1])
                    after_img_path = path.join(target_img_dir, after_img_name)

                    # 増幅した画像を保存
                    Image.fromarray(target_img_ar).save(after_img_path)
                    cnt_dic[label] += 1
                    continue


                # 増幅処理
                for aug_index, aug in enumerate(img_augmentors):

                    #画像増幅する
                    aug_img = aug.augment_images([target_img_ar])[0]


                    # 増幅した画像ファイル名
                    after_img_name = "{0:05d}_{1:02d}_{2:02d}{3}".format(origin_img_index, bb_index+1, aug_index+1, path.splitext(img_path)[1])
                    after_img_path = path.join(target_img_dir, after_img_name)

                    # 増幅した画像を保存
                    Image.fromarray(aug_img).save(after_img_path)

                    cnt_dic[label] += 1



    # 各ラベルごとのデータ数を出力
    train_cnt_all = 0
    val_cnt_all = 0
    test_cnt_all = 0
    for label, train_cnt, val_cnt, test_cnt in zip(train_cnt_dic.keys(), train_cnt_dic.values(), val_cnt_dic.values(), test_cnt_dic.values()):
        train_cnt_all += train_cnt
        val_cnt_all += val_cnt
        test_cnt_all += test_cnt
        print("label:\t{0}\n\ttrain_count:\t{1:d}\n\tval_count:\t{2:d}\n\ttest_count:\t{3:d}".format(label, train_cnt, val_cnt, test_cnt))

    print("tain data:\t{0:d}\nval data:\t{1:d}\ntest data:\t{1:d}".format(train_cnt_all, val_cnt_all, test_cnt_all))
    return {'train':train_cnt_all, 'val':val_cnt_all, 'test':test_cnt_all}

def download_file(url, filepath=''):
    """
    ファイルをダウンロードする
    """

    import os
    import urllib.request

    if filepath == '': filepath = os.path.basename(url)
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath)

def __format_class(origin_class, class_map):
    """
    クラスをclass_mapに従って変換する
    """
    if len(class_map) == 0: return origin_class

    for class_data in class_map:
        if origin_class in class_data['sub']:
            return class_data['label_id']

    raise RuntimeError()


def process_image_for_detection(validate_ratio, img_augmentors, img_edge_size, input_lst_path, input_img_root_path, output_root_path, class_map=[], test_ratio=0.0, validate_augment=False, test_augment=False, check_flg=False):
    """
    物体検出に使用する画像とlstデータの加工
    画像増幅と、検証/学習用へのデータ分けを行う。
    戻り値でtrainとvalのそれぞれのデータ数を返す。

    期待するclass_mapの形式の例：
    class_map=[{
        'label_id': 1,
        'label': 'screw',
        'sub': {
            '1' : 'nabeneji',
            '2' : 'saraneji',
            '3' : 'tyouneji',
    },{
        'label_id': 2,
        'label': 'spring',
        'sub': {'9' : 'hikibane'}
    },{
        'label_id': 3,
        'label': 'washer',
        'sub': {'8' : 'springwasher'}
    }]
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


    assert validate_ratio + test_ratio < 1.0, "validate_ratio +test_ratioは1以下のはず {} , {}".format(validate_ratio, test_ratio)


    #マージしたlstファイルと画像ファイルの出力先
    train_output_lst_path = path.join(output_root_path, "train.lst")
    train_output_img_root_path = path.join(output_root_path, "train")

    val_output_lst_path = path.join(output_root_path, "val.lst")
    val_output_img_root_path = path.join(output_root_path, "val")

    test_output_lst_path = path.join(output_root_path, "test.lst")
    test_output_img_root_path = path.join(output_root_path, "test")


    # 出力先をリセット
    if path.isdir(output_root_path):
        shutil.rmtree(output_root_path)

    os.makedirs(output_root_path)
    os.makedirs(train_output_img_root_path)
    os.makedirs(val_output_img_root_path)
    os.makedirs(test_output_img_root_path)

    if check_flg:
        check_output_img_root_path = path.join(output_root_path, "check")
        os.makedirs(check_output_img_root_path)


    train_output_lst = []
    val_output_lst = []
    test_output_lst = []
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
            target_img = np.array(resize_img_square(origin_img, img_edge_size))
            edge_correction = img_edge_size / max_edge

            # バウンディングボックスを生成
            bbs = []
            for bb_index in range(len(label_data)//label_width):
                bbs.append(ia.BoundingBox(
                    x1 = float(label_data[bb_index * label_width + 1]) * img_width * edge_correction,
                    y1 = float(label_data[bb_index * label_width + 2]) * img_height * edge_correction,
                    x2 = float(label_data[bb_index * label_width + 3]) * img_width * edge_correction,
                    y2 = float(label_data[bb_index * label_width + 4]) * img_height * edge_correction
                ))
            bbs_on_img = ia.BoundingBoxesOnImage(bbs, shape = target_img.shape)

            # 指定した確率で検証用データとして割り当てる
            random_num = random.random()
            if random_num < validate_ratio: # 検証用
                output_lst = val_output_lst
                output_img_root_path = val_output_img_root_path
                is_augment = validate_augment
            elif random_num < validate_ratio + test_ratio: # テスト用
                output_lst = test_output_lst
                output_img_root_path = test_output_img_root_path
                is_augment = test_augment
            else: # 学習用
                output_lst = train_output_lst
                output_img_root_path = train_output_img_root_path
                is_augment = True

            # 画像を増幅しない
            if not is_augment or len(img_augmentors) == 0:
                # そのまま画像を保存
                after_img_name = "{0:05d}_{1:03d}{2}".format(origin_img_index, 1, path.splitext(img_path)[1])
                after_img_path = path.join(output_img_root_path, after_img_name)
                Image.fromarray(target_img).save(after_img_path)

                # ラベルデータを上書き
                after_label_data = copy.deepcopy(label_data)
                for bb_index in range(len(label_data)//label_width):
                    after_label_data[bb_index * label_width] = str(__format_class(after_label_data[bb_index * label_width], class_map))
                    after_label_data[bb_index * label_width + 1] = str(bbs_on_img.bounding_boxes[bb_index].x1 /  img_edge_size)
                    after_label_data[bb_index * label_width + 2] = str(bbs_on_img.bounding_boxes[bb_index].y1 /  img_edge_size)
                    after_label_data[bb_index * label_width + 3] = str(bbs_on_img.bounding_boxes[bb_index].x2 /  img_edge_size)
                    after_label_data[bb_index * label_width + 4] = str(bbs_on_img.bounding_boxes[bb_index].y2 /  img_edge_size)


                # 画像用のlst形式のテキストを作成
                image_index = len(output_lst) + 1
                lst_dat = __create_lst(after_img_name, image_index, after_label_data)
                output_lst.append(lst_dat)
                continue




            #画像の増幅処理
            for aug_index, aug in enumerate(img_augmentors):
                # augmentorの変換方法を固定する(画像とバウンディングボックスそれぞれに対する変換方法を変えないようにするため)
                aug = aug.to_deterministic()

                #画像増幅する
                aug_img = aug.augment_images([target_img])[0]
                aug_bbs = aug.augment_bounding_boxes([bbs_on_img])[0]


                image_index = len(output_lst) + 1

                # 増幅した画像ファイル名
                after_img_name = "{0:05d}_{1:03d}{2}".format(origin_img_index, aug_index+1, path.splitext(img_path)[1])
                after_img_path = path.join(output_img_root_path, after_img_name)


                # 増幅した画像を保存
                Image.fromarray(aug_img).save(after_img_path)

                if check_flg:
                    after_bb_img = aug_bbs.draw_on_image(aug_img)
                    Image.fromarray(after_bb_img).save(path.join(check_output_img_root_path, after_img_name))



                # ラベルデータを上書き
                aug_label_data = copy.deepcopy(label_data)
                for bb_index in range(len(label_data)//label_width):
                    aug_label_data[bb_index * label_width] = str(__format_class(aug_label_data[bb_index * label_width], class_map))
                    aug_label_data[bb_index * label_width + 1] = str(aug_bbs.bounding_boxes[bb_index].x1 /  img_edge_size)
                    aug_label_data[bb_index * label_width + 2] = str(aug_bbs.bounding_boxes[bb_index].y1 /  img_edge_size)
                    aug_label_data[bb_index * label_width + 3] = str(aug_bbs.bounding_boxes[bb_index].x2 /  img_edge_size)
                    aug_label_data[bb_index * label_width + 4] = str(aug_bbs.bounding_boxes[bb_index].y2 /  img_edge_size)




                # 増幅画像用のlst形式のテキストを作成
                lst_dat = __create_lst(after_img_name, image_index, aug_label_data)

                output_lst.append(lst_dat)



    # 作成したデータを要素ごとに改行して書き出す
    with open(train_output_lst_path, 'w') as out_f:
        out_f.write('\n'.join(train_output_lst))

    if len(val_output_lst) > 0:
        with open(val_output_lst_path, 'w') as out_f:
            out_f.write('\n'.join(val_output_lst))
    if len(test_output_lst) > 0:
        with open(test_output_lst_path, 'w') as out_f:
            out_f.write('\n'.join(test_output_lst))

    train_cnt = len(train_output_lst)
    val_cnt = len(val_output_lst)
    test_cnt = len(test_output_lst)
    print("train data: ",train_cnt)
    print("validation data: ", val_cnt)
    print("test data: ", test_cnt)

    return {'train':train_cnt, 'val':val_cnt, 'test':test_cnt}



def create_recordio_and_upload_to_s3(input_img_path, s3_bucket_name, output_s3_obj_path):
    """
    画像ファイルからRecordIOを作成してS3へアップロードする
    """

    import subprocess
    import boto3
    from os import path

    # lstファイルとRecordIOを生成するツールをダウンロード
    script_path = './im2rec.py'
    download_file('https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py', script_path)


    # 出力するlstファイルの接頭辞
    out_prefix = path.basename(input_img_path)

    # lstファイル作成
    subprocess.run("python {0} --list --recursive {1} {2}".format(script_path, out_prefix, input_img_path), shell=True, check=True)

    # 分類用lstファイルのパス
    lst_path = out_prefix + '.lst'


    # RecordIOファイル作成
    subprocess.run("python {0} --num-thread 4 --pack-label {1} {2}".format(script_path, lst_path, input_img_path), shell=True, check=True)

    # recのファイル名
    rec_path = out_prefix + '.rec'

    # S3へアップロード
    s3 = boto3.resource('s3')
    s3.Bucket(s3_bucket_name).upload_file(rec_path, output_s3_obj_path)


def create_recordio_and_upload_to_s3_from_lst(input_img_path, input_lst_path, s3_bucket_name, output_s3_obj_path):
    """
    画像ファイルとlstファイルからRecordIOを作成してS3へアップロードする
    """

    import subprocess
    import boto3
    from os import path

    # lstファイルとRecordIOを生成するツールをダウンロード
    script_path = './im2rec.py'
    download_file('https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py', script_path)


    # RecordIOファイル作成
    subprocess.run("python {0} --num-thread 4 --pack-label {1} {2}".format(script_path, input_lst_path, input_img_path), shell=True, check=True)

    # recのファイル名
    rec_path = path.join(path.dirname(input_lst_path), path.splitext(path.basename(input_lst_path))[0] + '.rec')

    # S3へアップロード
    s3 = boto3.resource('s3')
    s3.Bucket(s3_bucket_name).upload_file(rec_path, output_s3_obj_path)



def __divide_decimal(num1, num2):
    """
    渡された数字をDecimal型として扱って、割り算する
    """
    from decimal import Decimal

    return (Decimal(num1) / Decimal(num2))


def __create_lst_from_pascalvoc(index, xml_path, sub_class_list, class_map):
    """
    pascalvoc形式のxmlファイルから画像のラベルデータを取得する
    """
    import xml.etree.ElementTree as ET
    from re import sub
    tree = ET.parse(xml_path)

    root = tree.getroot()
    img_name = root.find('filename').text

    # 画像内に複数のバウンディングボックスは存在しない(ただ一つだけのデータのみが対象)
    obj = root.find('object')
    bndbox = obj.find('bndbox')

    # 画像サイズ
    size = root.find('size')
    width = size.find('width').text
    height = size.find('height').text


    # そのラベルが属する親ラベル(例：カモメ→鳥)
    label = sub('_\d+.*', '', img_name)
    super_label = obj.find('name').text
    xmin = bndbox.find('xmin').text
    ymin = bndbox.find('ymin').text
    xmax = bndbox.find('xmax').text
    ymax = bndbox.find('ymax').text

    # 画像分類用クラスデータリストが存在するか調べる
    if label not in sub_class_list:
        sub_class_list[label] = len(sub_class_list)

    # マップデータに親クラスデータが存在するか調べる
    super_label_id = -1

    for super_cls_index, sup_cls_data in class_map.items():
        if sup_cls_data['label'] == super_label:
            super_label_id = super_cls_index
            break

    # 該当する親クラスが存在しないので、新規作成
    if super_label_id == -1:
        super_label_id = len(class_map)
        class_map[super_label_id] = {
            'label_id' : super_label_id,
            'label' : super_label,
            'sub' : {}
        }

    # サブクラスが存在していないのであれば、作成して入れる
    if label not in class_map[super_label_id]['sub'].values():
        class_map[super_label_id]['sub'][sub_class_list[label]] = label


    # lst形式のデータ作成をおこなう
    header_size = 2
    label_width = 5
    annotation_data = '\t'.join([
        str(sub_class_list[label]),
        str(__divide_decimal(xmin, width)),
        str(__divide_decimal(ymin, height)),
        str(__divide_decimal(xmax, width)),
        str(__divide_decimal(ymax, height))
    ])
    return ('\t'.join([
        str(index),
        str(header_size),
        str(label_width),
        annotation_data,
        img_name]), sub_class_list, class_map)


def create_lst_from_pascalvoc_dir(input_path, output_path):
    """
    対象のディレクトリに入ってるPascalVOC形式のxmlからlstファイルとクラスデータを作成する
    """
    import json
    import glob
    import os
    from os import path
    import shutil
    from tqdm import tqdm_notebook as tqdm

    if path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)


    # .lst形式のアノテーションファイルの中身をリストで作成する
    out_content_list = []
    class_map = {}
    sub_class_list = {}
    img_index = 1
    for file_path in tqdm(glob.glob(path.join(input_path, '*'))):
        if path.splitext(file_path)[1] != '.xml':
            continue

        # xmlを読み込んでlst形式のデータに変換。クラスデータも更新する
        lst_data, sub_class_list, class_map = __create_lst_from_pascalvoc(img_index, file_path, sub_class_list, class_map)
        img_index += 1
        out_content_list.append(lst_data)


    output_lst_path = path.join(output_path, 'lst.lst')
    output_class_path = path.join(output_path, 'class.json')

    # 作成したデータを要素ごとに改行して書き出す
    with open(output_lst_path, 'w') as out_f:
        out_f.write('\n'.join(out_content_list))

    with open(output_class_path, 'w') as out_f:
        class_data = {
            'class_list' : sub_class_list,
            'class_map' : class_map
        }
        json.dump(class_data, out_f)


def classify_img(pil_img, classes, classifier):
    """
    画像を分類
    クラス名と予測確率を返す。
    """
    import io
    import numpy as np
    from PIL import Image
    import time
    import imgmlutil # 自作のユーティリティパッケージ


    pil_img = imgmlutil.resize_img_square(pil_img, 224)

    # byte形式に変換
    img_byte = io.BytesIO()
    pil_img.save(img_byte, format='JPEG')
    img_byte = img_byte.getvalue()

    # 種類判別をするために画像分類器に投げる
    response = classifier.predict(img_byte)

    # 確率が高いものを選択
    class_id = np.argmax(response)
    class_name = str(class_id)
    if class_id < len(classes):
        class_name = classes[class_id]
    return class_name, response[class_id]





def classify_and_visualize_detection(img_file, dets, detection_class_map, classification_classes, classifier, thresh=0.6):
        """
        検出結果の可視化と分類
        検出結果データ(prediction_result['predictions'][0]['prediction'])をもとに検出部分を切り取り、画像分類する。
        その結果を可視化する。（バウンディングボックスと検出・分類結果のラベル）
        """
        import random
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from PIL import Image


        img=mpimg.imread(img_file)
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()

        #画像読み込む
        pil_img = Image.open(img_file)

        # 検出結果毎に処理していく(thresh未満の確率のものは飛ばす)
        for i, det in enumerate(dets):
            (klass, score, x0, y0, x1, y1) = det
            if score < thresh:
                continue
            cls_id = int(klass)
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())

            #検出位置
            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)


            class_name = str(cls_id)
            classification_prob = -1
            if detection_class_map and len(detection_class_map) > cls_id:

                # 検出部分を切り取って分類する
                class_name = detection_class_map[cls_id]['label']

                crop_img = pil_img.crop((xmin, ymin, xmax, ymax))
                classification_class_name, classification_prob = classify_img(crop_img, classification_classes, classifier)
                class_name += '_' + classification_class_name

            #  分類によって色を分ける
            if class_name not in colors:
                colors[class_name] = (random.random(), random.random(), random.random())

            # 枠を表示
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=3.5)
            plt.gca().add_patch(rect)

            # 名称と確率を表示 (名称 予測確率)
            label = '{:s} {:.3f}'.format(class_name, score)
            if classification_prob > -1:
                label += ' {:.3f}'.format(classification_prob)

            plt.gca().text(xmin, ymin - 2,
                            label,
                            bbox=dict(facecolor=colors[class_name], alpha=0.5),
                                    fontsize=12, color='white')
        # 図を出力
        plt.show()