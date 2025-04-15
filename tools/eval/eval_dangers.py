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


def roi_filter(anno_json, pred_json, roi_path, is_c_language = False):
    anno_data = load_json(anno_json)

    #获取各图像对应的roi路口
    img_name_id_map = {}
    for img_info in anno_data['images']:
        img_name = img_info['file_name'].split('/')[-1].split('_')
        if len(img_name) > 2:
            lukou = img_name[0].lower()+'_'+str(int(img_name[1])-704)+'.jpg'
        else:
            lukou = img_name[0]+'.jpg'
        img_name_id_map[img_info['id']] = lukou

    # 读取预测框
    if is_c_language:
        img_id_map = {}
        for img_info in anno_data['images']:
            img_id_map[img_info['file_name'].split('/')[-1]] = img_info['id']
        pred_data = []
        with open(pred_json, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.split(' ')
            if len(line) > 2:
                pred_info = {'image_id':img_id_map[line[0]],'score':float(line[-1]),'bbox':[float(line[7]),float(line[8]),float(line[9])-float(line[7]),float(line[10])-float(line[8])],'category_id':int(line[-2])}
                pred_data.append(pred_info)
    else:
        pred_data = load_json(pred_json)

    # 读取roi图像
    rois = {}
    roi_names = os.listdir(roi_path)
    for roi_name in roi_names:
        roi = cv2.imread(os.path.join(roi_path+roi_name))
        if len(roi.shape) == 3:
            roi = roi[:,:,0]
        roi = roi//255
        rois[roi_name] = roi

    # 对预测框进行roi过滤
    filter_pred = []
    for pred_info in pred_data:
        bbox = pred_info['bbox']
        area = bbox[2] * bbox[3]
        if img_name_id_map[pred_info['image_id']] in rois:
            roi = rois[img_name_id_map[pred_info['image_id']]]
            val_area = np.sum(roi[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])])
            if (val_area / area) > 0.8:
                filter_pred.append(pred_info)
        else:
            filter_pred.append(pred_info)
    
    #对标注框进行roi过滤
    filter_anno = {
                "categories": anno_data['categories'],
                "images": anno_data['images'],
                "annotations": []
    }

    for anno_info in anno_data["annotations"]:
        bbox = anno_info['bbox']
        area = bbox[2] * bbox[3]
        if img_name_id_map[anno_info['image_id']] in rois:
            roi = rois[img_name_id_map[anno_info['image_id']]]
            val_area = np.sum(roi[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])])
            if (val_area / area) > 0.8:
                filter_anno["annotations"].append(anno_info)
        else:
            filter_anno["annotations"].append(anno_info)

    return filter_pred, filter_anno

def without_roi_filter(anno_json, pred_json):
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

        dts = pred.loadAnns(pred.getAnnIds(imgIds=anno.getImgIds(), catIds=anno.getCatIds()))
        for dt in dts:
            if 0 < dt['category_id'] <= 3:
                dt['category_id'] = 1
            elif 3 < dt['category_id'] <=4:
                dt['category_id'] = 2
    else:
        for gt in gts:
            if 0 < gt['category_id'] <=2:
                gt['category_id'] = 1
            elif 2 < gt['category_id'] <=3:
                gt['category_id'] = 2
            elif 3 < gt['category_id'] <=4:
                gt['category_id'] = 3

        dts = pred.loadAnns(pred.getAnnIds(imgIds=anno.getImgIds(), catIds=anno.getCatIds()))
        for dt in dts:
            if 0 < dt['category_id'] <= 2:
                dt['category_id'] = 1
            elif 2 < dt['category_id'] <=3:
                dt['category_id'] = 2
            elif 3 < dt['category_id'] <=4:
                dt['category_id'] = 3
        
    eval = COCOeval(anno, pred, 'bbox')
    eval.params.iouThrs = np.linspace(.1, 0.9, int(np.round((0.9 - .1) / .1)) + 1, endpoint=True)

    eval.evaluate()
    eval.accumulate()
    # eval.summarize()

    iou_index = np.where(eval.params.iouThrs == iou_thr)[0]

    area_index = eval.params.areaRngLbl.index("all")
    maxdet_index = eval.params.maxDets.index(10)
    for recall_thr in recall_thrs:
        recall_thr = np.around(recall_thr, 2)
        recall_thrs_index = np.where(eval.params.recThrs == recall_thr)[0]
        precision = eval.eval["precision"][iou_index, recall_thrs_index, cat_ids, area_index, maxdet_index]
        score = eval.eval["scores"][iou_index, recall_thrs_index, cat_ids, area_index, maxdet_index]
        print('score: {}, recall: {}, precision: {}'.format(score, recall_thr, precision))
            

# 标注框和预测框可视化
def draw_bbox(score_thr, anno_json, pred_json, img_path, result_path):
    score_thr = {0:0, 1: 0.5, 2: 0.5, 3: 2, 4:2}
    # score_thr = {0:0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4:0}
    pred_json, anno_json = without_roi_filter(anno_json, pred_json)
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
        annId = pred.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)                  
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
            if score>score_thr[target['category_id']] and 0<=target['category_id']<=3:
                have_bbox = True
                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
                cv2.putText(img, str(target['category_id']) + " " + str(round(score,2)), (bbox[0]-5, bbox[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # cv2.putText(img, str(round(score,2)), (bbox[0]-5, bbox[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 250), 2)
        
        # 获得标注框信息
        ann_ids = anno.getAnnIds(imgIds=img_id)
        targets = anno.loadAnns(ann_ids)
        for target in targets:
            if 0<=target['category_id']<4:
                bbox=np.array(target['bbox'])
                bbox[bbox<0]=0
                bbox=bbox.astype(int)
                have_anno = True
                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
                cv2.putText(img, 'gt:'+str(target['category_id']), (bbox[0]-5, bbox[1]+bbox[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if have_bbox or have_anno:
            dirpath= os.path.dirname(result_path+path)
            os.makedirs(dirpath, exist_ok=True)
            cv2.imwrite(result_path+path,img)


def only_draw_bbox(score_thr, anno_json, pred_json, img_path, result_path):
    score_thr = {0:0, 1: 0.36, 2: 0.21, 3: 0.25, 4:0}
    pred_json, anno_json = without_roi_filter(anno_json, pred_json)
    anno = COCO(anno_json)
    
    # pred = anno.loadRes(pred_json)  # init predictions api
    # cats = pred.loadCats(pred.getCatIds()) 
    # cat_nms=[cat['name'] for cat in cats]
    # print('COCO categories: \n{}\n'.format(' '.join(cat_nms)))
    # print("{:<15} {:<5}     {:<10}".format('classname', 'imgnum', 'bboxnum'))
    # print('---------------------------------')

    # for cat_name in cat_nms:
    #     catId = pred.getCatIds(catNms=[cat_name])
    #     imgId = pred.getImgIds(catIds=catId)
    #     annId = pred.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)                  
    #     print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))

    ids = list(anno.imgs.keys())
    print("number of images: {}".format(len(ids)))
    for img_id in ids:
        have_bbox = False
        have_anno = False
        path = anno.loadImgs(img_id)[0]['file_name']
        img=cv2.imread(img_path+path)

        dirpath= os.path.dirname(result_path+path)
        os.makedirs(dirpath, exist_ok=True)
        cv2.imwrite(result_path+path,img)

        # cv2.imwrite(result_path+path,img)

        # # 获得预测框信息
        # ann_ids = pred.getAnnIds(imgIds=img_id)
        # targets = pred.loadAnns(ann_ids)
        # for target in targets:
        #     bbox=np.array(target['bbox'])
        #     bbox[bbox<0]=0
        #     bbox=bbox.astype(int)
        #     score = target['score']
        #     if score>score_thr[target['category_id']] and 0<target['category_id']<=3:
        #         have_bbox = True
        #         # img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 2)
        #         cv2.putText(img, str(target['category_id']), (bbox[0]-5, bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 250), 2)
        #         cv2.putText(img, str(round(score,2)), (bbox[0]-5, bbox[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 250), 2)
        
        # 获得标注框信息
        ann_ids = anno.getAnnIds(imgIds=img_id)
        targets = anno.loadAnns(ann_ids)
        for target in targets:
            bbox = np.array(target['bbox'])
            bbox[bbox < 0] = 0
            bbox = bbox.astype(int)
            have_anno = True
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, 'gt:' + str(target['category_id']), (bbox[0] - 5, bbox[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # if have_bbox or have_anno:
        with_box_path = result_path + path.replace('.jpg', '_with_box.jpg')
        cv2.imwrite(with_box_path, img)


if __name__ == '__main__':
    anno_json = '/root/paddlejob/workspace/env_run/lvfeng/output/抛洒物危险等级1227val/all_val_sel_1212.json'    # 标注框json路径
    pred_json = 'bbox.json'    # 预测框json路径
    roi_path = ''     # 道路roi路径

    # 计算p，r值 ， iou_thr：iou阈值 ， recall_thrs: recall值, cat_ids:要评测的类别
    recall_thrs = np.linspace(0.9, 0.6, int(np.round((0.9 - 0.6) / 0.05)) + 1, endpoint=True)  
    compute_pr(iou_thr = 0.1, recall_thrs=recall_thrs, cat_ids = [0, 1, 2], anno_json = anno_json, pred_json = pred_json, roi_path=roi_path, is_c_language=False, merge_cate=True)
    compute_pr(iou_thr = 0.1, recall_thrs=recall_thrs, cat_ids = [0, 1, 2], anno_json = anno_json, pred_json = pred_json, roi_path=roi_path, is_c_language=False, merge_cate=False)

    # # 预测结果可视化
    # img_path = '/root/paddlejob/workspace/env_run/Anomaly/DATA/paosawu_danger/images/'
    # result_path = '/root/paddlejob/workspace/env_run/Anomaly/ppyoloe/yy_vis/'
    # draw_bbox(0.1, anno_json, pred_json, img_path, result_path)
    # only_draw_bbox(0.1, anno_json, pred_json, img_path, result_path)