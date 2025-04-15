from unittest import result
from coco import COCO
from cocoeval import COCOeval
import json
import numpy as np
import cv2
import os
from numpy import *

def load_json(file):
    with open(file, 'r') as fp:
        json_obj = json.load(fp)
    return json_obj


def load_pred(dirs, imgs):
    preds = []
    name2id = {}
    for im in imgs:
        name2id[im['file_name']] = im['id']
    
    files = os.listdir(dirs)
    for file in files:
        file = os.path.join(dirs, file)
        file_name = file.split('/')[-1].split('.')[0]

        with open(file, 'r') as fp:
            lines = fp.readlines()
        
        for line in lines:
            if "even" not in line:
                continue
            name = file_name + "/" + line.split(' event ')[0]
            id = name2id[name]
            score = float(line.split(' ')[-1])
            cat_id = int(line.split(' ')[-2]) + 1
            xmin = float(line.split(' ')[-6])
            ymin = float(line.split(' ')[-5])
            w = float(line.split(' ')[-4]) - float(line.split(' ')[-6])
            h = float(line.split(' ')[-3]) - float(line.split(' ')[-5])

            itm = {
                'image_id': id,
                'category_id': cat_id,
                'bbox': [xmin, ymin, w, h],
                'score': score
            }
            preds.append(itm)
    json.dump(preds, open('bbox_fc.json', 'w'))
    return preds


def nms(cases):
    thresh = 0.5
    scores = []
    bboxes = []
    for case in cases:
        scores.append(float(case[15]))
        bboxes.append(list(map(float, case[4:8])))
    scores = np.array(scores)
    bboxes = np.array(bboxes)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    temp = []
    while order.size > 0:
        i = order[0]
        temp.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return temp


def calc_iou(bbox1, bbox2):
    if not (bbox1 and bbox2):
        return []
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, w1, h1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, w2, h2, = np.split(bbox2, 4, axis=-1)

    area1 = w1 * h1
    area2 = w2 * h2

    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymin1+h1, np.squeeze(ymin2+h2, axis=-1))
    xmax = np.minimum(xmin1+w1, np.squeeze(xmin2+w2, axis=-1))

    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w

    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union


def without_roi_filter(anno_json, pred_json):
    anno_data = load_json(anno_json)
    filter_pred = load_pred(pred_json, anno_data["images"])
    return filter_pred, anno_data

def without_roi_filter_fc(anno_json, pred_json):
    anno_data = load_json(anno_json)
    filter_pred = load_json(pred_json)
    return filter_pred, anno_data

def compute_pr(iou_thr, recall_thrs, cat_ids, anno_json, pred_json, roi_path, is_c_language, merge_cate = False):
    '''
    修改：
    1. 标注gt面积小于24*16的忽略
    2. 对于沙土类别,iou分母不是两area的并,是min(area1,area2)
    3. 将路面roi外的框过滤掉
    '''
    # filter_pred,filter_anno = roi_filter(anno_json, pred_json, roi_path, is_c_language)
    filter_pred, filter_anno = without_roi_filter(anno_json, pred_json)
    anno = COCO(filter_anno)    # init annotations api
    print("number of images anno: {}".format(len(anno.imgs.keys())))
    pred = anno.loadRes(filter_pred)  # init predictions api
    dts = pred.loadAnns(pred.getAnnIds(imgIds=pred.getImgIds(), catIds=pred.getCatIds()))
    P0_num = 0
    P1_num = 0
    P2_num = 0
    for gt in dts:
        if gt['category_id'] == 1:
            P0_num += 1
        elif gt['category_id'] == 2:
            P1_num += 1
        elif gt['category_id'] == 3:
            P2_num += 1
    print('P0_num: {}, P1_num: {}, P2_num: {}'.format(P0_num, P1_num, P2_num))

    print("number of images pred: {}".format(len(pred.imgs.keys())))
    gts=anno.loadAnns(anno.getAnnIds(imgIds=anno.getImgIds(), catIds=anno.getCatIds()))

    P0_num = 0
    P1_num = 0
    P2_num = 0
    for gt in gts:
        if gt['category_id'] == 1:
            P0_num += 1
        elif gt['category_id'] == 2:
            P1_num += 1
        elif gt['category_id'] == 3:
            P2_num += 1
    print('P0_num: {}, P1_num: {}, P2_num: {}'.format(P0_num, P1_num, P2_num))

    # 3-8类合为一类评测
    if merge_cate:
        for gt in gts:
            if 0 < gt['category_id'] <=3:
                gt['category_id'] = 1
            elif 3 < gt['category_id'] <=4:
                gt['category_id'] = 2

        for dt in dts:
            if dt['category_id'] == 1:
                if dt['score'] > 0.1:
                    dt['score'] = 0.1
                else:
                    dt['score'] = 0.
            elif dt['category_id'] == 2:
                if dt['score'] > 0.1:
                    dt['score'] = 0.1
                else:
                    dt['score'] = 0.
            elif dt['category_id'] == 3:
                if dt['score'] > 0.1:
                    dt['score'] = 0.1
                else:
                    dt['score'] = 0.

        for dt in dts:
            if 0 < dt['category_id'] <= 3:
                dt['category_id'] = 1
            else:
                dt['category_id'] = 2

    eval = COCOeval(anno, pred, 'bbox')
    eval.params.iouThrs = np.linspace(.1, 0.9, int(np.round((0.9 - .1) / .1)) + 1, endpoint=True)

    eval.evaluate()
    eval.accumulate()
    # eval.summarize()

    iou_index = np.where(eval.params.iouThrs == iou_thr)[0]

    area_index = eval.params.areaRngLbl.index("all")
    maxdet_index = eval.params.maxDets.index(100)
    for recall_thr in recall_thrs:
        recall_thr = np.around(recall_thr, 2)
        recall_thrs_index = np.where(eval.params.recThrs == recall_thr)[0]
        precision = eval.eval["precision"][iou_index, recall_thrs_index, cat_ids, area_index, maxdet_index]
        score = eval.eval["scores"][iou_index, recall_thrs_index, cat_ids, area_index, maxdet_index]
        print('score: {}, recall: {}, precision: {}'.format(score, recall_thr, precision))
            

# 标注框和预测框可视化
def draw_bbox(score_thr, anno_json, pred_json, img_path, result_path):
    # score_thr = {0:0, 1: 0.1535, 2: 0.3488, 3: 0.5198, 4:0}
    score_thr = {0:0, 1: 0.1, 2: 0.1, 3: 0.1, 4:2}
    pred_json, anno_json = without_roi_filter_fc(anno_json, pred_json)
    anno = COCO(anno_json)
    
    pred = anno.loadRes(pred_json)  # init predictions api
    cats = pred.loadCats(pred.getCatIds()) 
    cat_nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(cat_nms)))
    print("{:<15} {:<5}     {:<10}".format('classname', 'imgnum', 'bboxnum'))
    print('---------------------------------')

    for cat_name in cat_nms:
        catId = pred.getCatIds(catNms=[cat_name])
        imgId = pred.getImgIds(catIds=catId)
        annId = pred.getAnnIds(imgIds=imgId, catIds=catId)
        print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))

    for cat_name in cat_nms:
        catId = anno.getCatIds(catNms=[cat_name])
        imgId = anno.getImgIds(catIds=catId)
        annId = anno.getAnnIds(imgIds=imgId, catIds=catId)                  
        print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))

    ids = list(anno.imgs.keys())
    print("number of images: {}".format(len(ids)))
    for img_id in ids:
        have_bbox = False
        have_anno = False
        path = anno.loadImgs(img_id)[0]['file_name']
        img=cv2.imread(img_path+path)
        # cv2.imwrite(result_path+path,img)

        # 获得预测框信息
        ann_ids = pred.getAnnIds(imgIds=img_id)
        targets = pred.loadAnns(ann_ids)
        for target in targets:
            bbox=np.array(target['bbox'])
            bbox[bbox<0]=0
            bbox=bbox.astype(int)
            score = target['score']
            area = bbox[2]*bbox[3]
            if score>score_thr[target['category_id']] and 0<target['category_id']<=3:
                have_bbox = True
                # img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 2)
                cv2.putText(img, str(target['category_id']) + " " + str(round(score,2)), (bbox[0]-5, bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 250), 2)
                # cv2.putText(img, str(target['category_id']), (bbox[0]-5, bbox[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # cv2.putText(img, str(round(score,2)), (bbox[0]-5, bbox[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 250), 2)
        
        # 获得标注框信息
        ann_ids = anno.getAnnIds(imgIds=img_id)
        targets = anno.loadAnns(ann_ids)
        for target in targets:
            if 0<target['category_id']<4:
                bbox=np.array(target['bbox'])
                bbox[bbox<0]=0
                bbox=bbox.astype(int)
                have_anno = True
                area = bbox[2]*bbox[3]
                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
                cv2.putText(img, 'gt:'+str(target['category_id']), (bbox[0]-5, bbox[1]+bbox[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # if have_bbox or have_anno:
        dirpath= os.path.dirname(result_path+path)
        os.makedirs(dirpath, exist_ok=True)
        cv2.imwrite(result_path+path,img)


if __name__ == '__main__':
    anno_json = '/root/paddlejob/workspace/env_run/Anomaly/DATA/coco/all_val_sel_1212_with_badcase.json'    # 标注框json路径
    results_path = '/root/paddlejob/workspace/env_run/Anomaly/Sand/ppyoloe/results'  # 预测框json路径
    pred_json =  '/root/paddlejob/workspace/env_run/Anomaly/Sand/ppyoloe/bbox_fc.json'

    # 计算p，r值 ， iou_thr：iou阈值 ， recall_thrs: recall值, cat_ids:要评测的类别
    recall_thrs = np.linspace(1.0, 0.1, int(np.round((1.0 - 0.1) / 0.1)) + 1, endpoint=True)

    compute_pr(iou_thr = 0.1, recall_thrs=recall_thrs, cat_ids = [0], anno_json = anno_json, pred_json = results_path, roi_path="", is_c_language=False, merge_cate=True)
    # compute_pr(iou_thr = 0.1, recall_thrs=recall_thrs, cat_ids = [0, 1, 2], anno_json = anno_json, pred_json = results_path, roi_path="", is_c_language=False, merge_cate=False)

    # 预测结果可视化
    img_path = '/root/paddlejob/workspace/env_run/Anomaly/DATA/coco/'
    result_path = '/root/paddlejob/workspace/env_run/Anomaly/Sand/ppyoloe/vis_pred/'
    draw_bbox(0.1, anno_json, pred_json, img_path, result_path)