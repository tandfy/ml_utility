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
    from tqdm import tqdm

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
        return pil_img.resize(edge_size, edge_size)
    else:
        square_img = Image.new(pil_img.mode, (longer_edge, longer_edge), background_color)
        # 正方形の背景画像の左上に画像を配置
        square_img.paste(pil_img, (0,0))
        return square_img.resize((edge_size, edge_size))

def process_image_for_classification(validate_ratio, img_augmentors, img_edge_size, input_lst_path, input_img_root_path, output_root_path, except_class_list=[]):
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
    from tqdm import tqdm
    import numpy as np

    assert validate_ratio < 1.0, "validate_ratio は1以下のはず" + str(validate_ratio)

    #マージした画像ファイルの出力先
    train_output_img_root_path = path.join(output_root_path, "train")

    val_output_img_root_path = path.join(output_root_path, "val")


    # 出力先をリセット
    if path.isdir(output_root_path):
        shutil.rmtree(output_root_path)

    os.makedirs(output_root_path)
    os.makedirs(train_output_img_root_path)
    os.makedirs(val_output_img_root_path)

    val_cnt_dic = {}
    train_cnt_dic = {}
    with open(input_lst_path) as lst_f:
        for line in tqdm(lst_f.readlines()):
            line = line.strip()
            if not line: continue

            #lst形式のデータを読み取って、変数に入れる
            origin_img_index, header_size, label_width, label_data, img_path = __read_lst(line)
            img_path = path.join(input_img_root_path, img_path)

            # 画像を読み込む
            origin_img = Image.open(img_path)

            # バウンディングボックスから該当オブジェクトを切り取る
            img_height = origin_img.width
            img_width = origin_img.height
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
                    train_cnt_dic[label] = 0

                # 指定した確率で検証用データとして割り当てる
                if random.random() < validate_ratio:
                    output_img_root_path = val_output_img_root_path
                    val_cnt_dic[label] += aug_num
                else:
                    output_img_root_path = train_output_img_root_path
                    train_cnt_dic[label] += aug_num

                # 保存先の確認
                target_img_dir = path.join(output_img_root_path, label)
                if not path.isdir(target_img_dir):
                    os.makedirs(target_img_dir)


                #画像
                for aug_index, aug in enumerate(img_augmentors):

                    #画像増幅する
                    aug_img = aug.augment_images([target_img_ar])[0]


                    # 増幅した画像ファイル名
                    after_img_name = "{0:05d}_{1:02d}_{2:02d}{3}".format(origin_img_index, bb_index+1, aug_index+1, path.splitext(img_path)[1])
                    after_img_path = path.join(target_img_dir, after_img_name)

                    # 増幅した画像を保存
                    Image.fromarray(aug_img).save(after_img_path)


    # 各ラベルごとのデータ数を出力
    train_cnt_all = 0
    val_cnt_all = 0
    for label, train_cnt, val_cnt in zip(train_cnt_dic.keys(), train_cnt_dic.values(), val_cnt_dic.values()):
        train_cnt_all += train_cnt
        val_cnt_all += val_cnt
        print("label:\t{0}\n\ttrain_count:\t{1:d}\n\tval_count:\t{2:d}".format(label, train_cnt, val_cnt))

    print("tain data:\t{0:d}\nval data:\t{1:d}".format(train_cnt_all, val_cnt_all))
    return {'train':train_cnt_all, 'val':val_cnt_all}

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

    for class_data in class_map:
        if origin_class in class_data['sub']:
            return class_data['label_id']

    raise RuntimeError()


def process_image_for_detection(validate_ratio, img_augmentors, img_edge_size, input_lst_path, input_img_root_path, output_root_path, class_map):
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
    from tqdm import tqdm


    assert validate_ratio < 1.0, "validate_ratio は1以下のはず" + str(validate_ratio)

    #マージしたlstファイルと画像ファイルの出力先
    train_output_lst_path = path.join(output_root_path, "train.lst")
    train_output_img_root_path = path.join(output_root_path, "train")

    val_output_lst_path = path.join(output_root_path, "val.lst")
    val_output_img_root_path = path.join(output_root_path, "val")


    # 出力先をリセット
    if path.isdir(output_root_path):
        shutil.rmtree(output_root_path)

    os.makedirs(output_root_path)
    os.makedirs(train_output_img_root_path)
    os.makedirs(val_output_img_root_path)

    train_output_lst = []
    val_output_lst = []
    with open(input_lst_path) as lst_f:
        for line in tqdm(lst_f.readlines()):
            line = line.strip()
            if not line: continue

            #lst形式のデータを読み取って、変数に入れる
            origin_img_index, header_size, label_width, label_data, img_path = __read_lst(line)
            img_path = path.join(input_img_root_path, img_path)

            # 画像を読み込む
            target_img = np.array(resize_img_square(Image.open(img_path), img_edge_size))

            # バウンディングボックスを生成
            img_height = target_img.shape[0]
            img_width = target_img.shape[1]
            bbs = []
            for bb_index in range(len(label_data)//label_width):
                bbs.append(ia.BoundingBox(
                    x1 = float(label_data[bb_index * label_width + 1]) * img_width,
                    y1 = float(label_data[bb_index * label_width + 2]) * img_height,
                    x2 = float(label_data[bb_index * label_width + 3]) * img_width,
                    y2 = float(label_data[bb_index * label_width + 4]) * img_height
                ))
            bbs_on_img = ia.BoundingBoxesOnImage(bbs, shape = target_img.shape)

            # 指定した確率で検証用データとして割り当てる
            if random.random() < validate_ratio:
                output_lst = val_output_lst
                output_img_root_path = val_output_img_root_path
            else:
                output_lst = train_output_lst
                output_img_root_path = train_output_img_root_path


            #画像
            for aug_index, aug in enumerate(img_augmentors):

                #画像増幅する
                aug_img = aug.augment_images([target_img])[0]
                aug_bbs = aug.augment_bounding_boxes([bbs_on_img])[0]


                image_index = len(output_lst) + 1

                # 増幅した画像ファイル名
                after_img_name = "{0:05d}_{1:03d}{2}".format(origin_img_index, aug_index+1, path.splitext(img_path)[1])
                after_img_path = path.join(output_img_root_path, after_img_name)


                # 増幅した画像を保存
                Image.fromarray(aug_img).save(after_img_path)

                # ラベルデータを上書き
                aug_label_data = copy.deepcopy(label_data)
                for bb_index in range(len(label_data)//label_width):
                    aug_label_data[bb_index * label_width] = str(__format_class(aug_label_data[bb_index * label_width], class_map))
                    aug_label_data[bb_index * label_width + 1] = str(aug_bbs.bounding_boxes[bb_index].x1 /  img_width)
                    aug_label_data[bb_index * label_width + 2] = str(aug_bbs.bounding_boxes[bb_index].y1 /  img_height)
                    aug_label_data[bb_index * label_width + 3] = str(aug_bbs.bounding_boxes[bb_index].x2 /  img_width)
                    aug_label_data[bb_index * label_width + 4] = str(aug_bbs.bounding_boxes[bb_index].y2 /  img_height)


                # 増幅画像用のlst形式のテキストを作成
                lst_dat = __create_lst(after_img_name, image_index, aug_label_data)

                output_lst.append(lst_dat)



    # 作成したデータを要素ごとに改行して書き出す
    with open(train_output_lst_path, 'w') as out_f:
        out_f.write('\n'.join(train_output_lst))

    if len(val_output_lst) > 0:
        with open(val_output_lst_path, 'w') as out_f:
            out_f.write('\n'.join(val_output_lst))

    train_cnt = len(train_output_lst)
    val_cnt = len(val_output_lst)
    print("train data: ",train_cnt)
    print("validation data: ", val_cnt)

    return {'train':train_cnt, 'val':val_cnt}



def create_recordio_and_upload_to_s3(input_img_path, s3_bucket_name, output_s3_obj_path):
    """
    画像ファイルからRecordIOを作成してS3へアップロードする
    """

    import subprocess
    import boto3

    # lstファイルとRecordIOを生成するツールをダウンロード
    script_path = './im2rec.py'
    download_file('https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py', script_path)


    # 出力するlstファイルの接頭辞
    lst_out_prefix = 'train'

    # lstファイル作成
    subprocess.run("python {0} --list --recursive {1} {2}".format(script_path, lst_out_prefix, input_img_path), shell=True, check=True)

    # 分類用lstファイルのパス
    lst_path = lst_out_prefix + '.lst'


    # RecordIOファイル作成
    subprocess.run("python {0} --num-thread 4 --pack-label {1} {2}".format(script_path, lst_path, input_img_path), shell=True, check=True)

    # recのファイル名
    rec_path = lst_out_prefix + '.rec'

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

