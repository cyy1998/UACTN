# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import time
import torch
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
parser = argparse.ArgumentParser("Retrieval Evaluation")

parser.add_argument('--class-sorting-file', type=str, default='../class_sorting/sketch_class_sorting.mat',
                    help="class sorting  flie of test sketches, .mat file")

parser.add_argument('--distance-type', type=str, choices=['cosine','euclidean'],default='cosine')
# parser.add_argument('--num-testsketch-samples', type=int, default=90*30)
# parser.add_argument('--num-classes', type=int, default=90)
# parser.add_argument('--num-view-samples', type=int, default=1258)
parser.add_argument('--num-testsketch-samples', type=int, default=171*30)
parser.add_argument('--num-classes', type=int, default=171)
parser.add_argument('--num-view-samples', type=int, default=8987)
# parser.add_argument('--num-testsketch-samples', type=int, default=18*30)
# parser.add_argument('--num-classes', type=int, default=15)
# parser.add_argument('--num-view-samples', type=int, default=300)
# parser.add_argument('--num-view-samples', type=int, default=80*13+79+64)

parser.add_argument('--test-sketch-feat-file', type=str,
                    default='./extract_features/alex_L2_test_sketch_feature.mat',
                    help="features flie of test sketches, .mat file")
parser.add_argument('--view-feat-flie', type=str,
                    default='./extract_features/alex_L2_test_view_feature.mat',
                    help="features flie of view images of 3d models, .mat file")

args = parser.parse_args()


def get_feat_and_labels(test_sketch_feat_file, view_feat_flie):
    """" read the features and labels of sketches and 3D models
    Args:
        test_sketch_feat_file: features flie of test sketches, it is .mat file
        view_feat_flie: features flie of view images of 3d models
    """
    sket_data_features = torch.load(test_sketch_feat_file)
    view_data_features1 = torch.load(view_feat_flie)


    sketch_feature = sket_data_features['sketch_feature']
    sketch_label = sket_data_features['sketch_labels']
    sketch_predict_label = sket_data_features['predict_label']
    """
    sketch_feature = sket_data_features['view_feature']
    print(sketch_feature.shape)
    sketch_label = sket_data_features['view_labels']
    """

    view_feature = view_data_features1['view_feature']
    view_label = view_data_features1['view_labels']


    return sketch_feature, sketch_label,sketch_predict_label, view_feature, view_label

def cal_euc_distance(sketch_feat,view_feat):
    distance_matrix = pairwise_distances(sketch_feat,view_feat)

    return distance_matrix

def cal_cosine_distance(sketch_feat,view_feat):
    distance_matrix = cosine_similarity(sketch_feat,view_feat)

    return distance_matrix

def evaluation_metric(distance_matrix, sketch_label, view_label,dist_type):
    """ calculate the evaluation metric

    Return:
        Av_NN:the precision of top 1 retrieval list
        Av_FT:Assume there are C relavant models in the database,FT is the
        recall of the top C-1 retrieval list
        Av_ST: recall of the top 2(C-1) retrieval list
        Av_E:the retrieval performance of the top 32 model in a retrieval list
        Av_DCG:normalized summed weight value related to the positions of related models
        Av_Precision:mAP1

    """
    from collections import Counter
    np.set_printoptions(suppress=True)
    index_label = np.zeros((view_label.shape[0],))
    # Get the number of samples for each category of 3D models
    view_label_count = {}
    view_label_list = list(np.reshape(view_label, (args.num_view_samples,)))
    view_label_set = set(view_label_list)
    count = 0
    for i in view_label_set:
        view_label_count[i] = view_label_list.count(i)
        #print(np.arange(view_label_count[i]))
        index_label[count:count+view_label_count[i]] = np.arange(view_label_count[i])
        #print(index_label[0:315])
        count+=view_label_count[i]
    #print(view_label_count)
    sketch_num = args.num_testsketch_samples
    view_num = args.num_view_samples

    P_points = np.zeros((sketch_num, 632));
    Av_Precision = np.zeros((sketch_num, 1));
    Av_NN = np.zeros((sketch_num, 1));
    Av_FT = np.zeros((sketch_num, 1));
    Av_ST = np.zeros((sketch_num, 1));
    Av_E = np.zeros((sketch_num, 1));
    Av_DCG = np.zeros((sketch_num, 1));

    for j in range(sketch_num):
        true_label = sketch_label[j]
        #print(true_label)

        view_label_num = view_label_count[true_label[0]]
        # print(view_label_num)
        dist_sort_index = np.zeros((args.num_view_samples, 1), dtype=int)
        count = 0
        if dist_type == 'euclidean':
            dist_sort_index = np.argsort(distance_matrix[j], axis=0)
        elif dist_type == 'cosine':
            dist_sort_index = np.argsort(-distance_matrix[j],axis = 0)
        dist_sort_index = np.reshape(dist_sort_index, (args.num_view_samples,))

        view_label_sort = view_label[dist_sort_index]
        index_label_sort = index_label[dist_sort_index]
        #print(view_label_sort)

        b = np.array([[0]])
        view_label_sort = np.insert(view_label_sort, 0, values=b, axis=0)

        G = np.zeros((view_num + 1, 1))
        for i in range(1, view_num + 1):
            if true_label == view_label_sort[i]:
                G[i] = 1
        G_sum = G.cumsum(0)

        NN = G[1]
        FT = G_sum[view_label_num] / view_label_num
        ST = G_sum[2 * view_label_num] / view_label_num

        P_32 = G_sum[32] / 32
        R_32 = G_sum[32] / view_label_num
        if (P_32 == 0) and (R_32 == 0):
            Av_E[j] = 0
        else:
            Av_E[j] = 2 * P_32 * R_32 / (P_32 + R_32)

        # 计算DCG
        NORM_VALUE = 1 + np.sum(1. / np.log2(np.arange(2, view_label_num + 1)))

        m = 1. / np.log2(np.arange(2, view_num + 1))
        m = np.reshape(m, [m.shape[0], 1])

        dcg_i = m * G[2:]
        dcg_i = np.vstack((G[1], dcg_i))
        Av_DCG[j] = np.sum(dcg_i) / NORM_VALUE;

        R_points = np.zeros((view_label_num + 1, 1), dtype=int)

        for n in range(1, view_label_num + 1):
            for k in range(1, view_num + 1):
                if G_sum[k] == n:
                    R_points[n] = k
                    break

        R_points_reshape = np.reshape(R_points, (view_label_num + 1,))

        P_points[j, 0:view_label_num] = np.reshape(G_sum[R_points_reshape[1:]] / R_points[1:], (view_label_num,))

        Av_Precision[j] = np.mean(P_points[j, 0:view_label_num])
        Av_NN[j] = NN
        Av_FT[j] = FT
        Av_ST[j] = ST
        #print(Av_Precision[j])

        #if Av_Precision[j] <=0.99:
            #print(j)
            #print(Av_Precision[j])
            #print("++++++++++++++++++++++++++++")
            #time.sleep(1)
        if j % 100 == 0:
            print("==> test samplses [%d/%d]" % (j, args.num_testsketch_samples))

    return Av_NN, Av_FT, Av_ST, Av_E, Av_DCG, Av_Precision

def main():
    MODE='a'
    sketch_feature, sketch_label,sketch_predict_label, view_feature, view_label = get_feat_and_labels(args.test_sketch_feat_file,
                                                                                 args.view_feat_flie)
    #print(view_label.shape)
    if args.distance_type == 'euclidean':
        distance_matrix = cal_euc_distance(sketch_feature,view_feature)
    elif args.distance_type == 'cosine':
        distance_matrix = cal_cosine_distance(sketch_feature,view_feature)

    if MODE=='CLF':
        for i in range(distance_matrix.shape[0]):
            distance_matrix[i,np.where(view_label==sketch_predict_label[i])[0]]+=1000
    elif MODE=='UP':
        for i in range(distance_matrix.shape[0]):
            distance_matrix[i,np.where(view_label==sketch_label[i])[0]]+=1000


    distance_matrix_data = {"distance_matrix":distance_matrix}
    torch.save(distance_matrix_data,"distance_matrix.mat")

    Av_NN, Av_FT, Av_ST, Av_E, Av_DCG, Av_Precision = evaluation_metric(distance_matrix,
                                                                        sketch_label, view_label,args.distance_type)

    torch.save(Av_Precision,'precison.mat')
    print("NN:", Av_NN.mean())
    print("FT:", Av_FT.mean())
    print("ST:", Av_ST.mean())
    print("E:", Av_E.mean())
    print("DCG:", Av_DCG.mean())
    print("mAP", Av_Precision.mean())


if __name__ == '__main__':
    main()