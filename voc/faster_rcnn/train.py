import keras
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from nets.frcnn import get_model
from nets.frcnn_training import (ProposalTargetCreator, classifier_cls_loss,
                                 classifier_smooth_l1, rpn_cls_loss,
                                 rpn_smooth_l1)
from utils.anchors import get_anchors
from utils.callbacks import LossHistory
from utils.dataloader import FRCNNDatasets
from utils.utils import get_classes
from utils.utils_bbox import BBoxUtility
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    classes_path    = 'model_data/voc_classes.txt'
    
    model_path      = 'model_data/resnet50_weights.h5'
    #------------------------------------------------------#
    #   输入的shape大小 
    #------------------------------------------------------#
    input_shape     = [600, 600]
    #---------------------------------------------#
    #   vgg或者resnet50
    #---------------------------------------------#
    backbone        = "resnet50"
    #------------------------------------------------------------------------#
    #   anchors_size用于设定先验框的大小，每个特征点均存在9个先验框。
    #   anchors_size每个数对应3个先验框。
    #   当anchors_size = [8, 16, 32]的时候，生成的先验框宽高约为：
    #   [128, 128] ; [256, 256]; [512, 512]; [128, 256]; 
    #   [256, 512]; [512, 1024]; [256, 128] ; [512, 256]; 
    #   [1024, 512]; 详情查看anchors.py
    #   如果想要检测小物体，可以减小anchors_size靠前的数。
    #   比如设置anchors_size = [64, 256, 512]
    #------------------------------------------------------------------------#
    anchors_size    = [128, 256, 512]

    #----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #----------------------------------------------------#
    #----------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 20
    Freeze_batch_size   = 8
    Freeze_lr           = 1e-4
    #----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #----------------------------------------------------#
    UnFreeze_Epoch      = 80
    Unfreeze_batch_size = 4
    Unfreeze_lr         = 1e-5
    #------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    #------------------------------------------------------#
    Freeze_Train        = True
    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    num_classes += 1
    anchors = get_anchors(input_shape, backbone, anchors_size)

    K.clear_session()
    model_rpn, model_all = get_model(num_classes, backbone = backbone)
    if model_path != '':
        #------------------------------------------------------#
        #   载入预训练权重
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        model_rpn.load_weights(model_path, by_name=True)
        model_all.load_weights(model_path, by_name=True)

    #--------------------------------------------#
    #   训练参数的设置
    #--------------------------------------------#
    callback        = TensorBoard(log_dir="logs")
    callback.set_model(model_all)
    loss_history    = LossHistory("logs/")

    bbox_util       = BBoxUtility(num_classes)
    roi_helper      = ProposalTargetCreator(num_classes)
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path,encoding='UTF8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path,encoding='UTF8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    freeze_layers = {'vgg' : 17, 'resnet50' : 141}[backbone]
    if Freeze_Train:
        for i in range(freeze_layers): 
            if type(model_all.layers[i]) != keras.layers.BatchNormalization:
                model_all.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_all.layers)))

    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        model_rpn.compile(
            loss = {
                'classification': rpn_cls_loss(),
                'regression'    : rpn_smooth_l1()
            }, optimizer = Adam(lr=lr)
        )
        model_all.compile(
            loss = {
                'classification'                        : rpn_cls_loss(),
                'regression'                            : rpn_smooth_l1(),
                'dense_class_{}'.format(num_classes)    : classifier_cls_loss(),
                'dense_regress_{}'.format(num_classes)  : classifier_smooth_l1(num_classes - 1)
            }, 
            metrics={'dense_class_{}'.format(num_classes): 'accuracy'},
            optimizer = Adam(lr=lr)
        )

        gen     = FRCNNDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train = True).generate()
        gen_val = FRCNNDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train = False).generate()

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch,
                    anchors, bbox_util, roi_helper)
            lr = lr*0.96
            K.set_value(model_rpn.optimizer.lr, lr)
            K.set_value(model_all.optimizer.lr, lr)

    if Freeze_Train:
        for i in range(freeze_layers): 
            if type(model_all.layers[i]) != keras.layers.BatchNormalization:
                model_all.layers[i].trainable = True

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        model_rpn.compile(
            loss = {
                'classification': rpn_cls_loss(),
                'regression'    : rpn_smooth_l1()
            }, optimizer = Adam(lr=lr)
        )
        model_all.compile(
            loss = {
                'classification'                        : rpn_cls_loss(),
                'regression'                            : rpn_smooth_l1(),
                'dense_class_{}'.format(num_classes)    : classifier_cls_loss(),
                'dense_regress_{}'.format(num_classes)  : classifier_smooth_l1(num_classes - 1)
            }, optimizer = Adam(lr=lr)
        )

        gen     = FRCNNDatasets(train_lines, input_shape, anchors, batch_size, num_classes, train = True).generate()
        gen_val = FRCNNDatasets(val_lines, input_shape, anchors, batch_size, num_classes, train = False).generate()

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_rpn, model_all, loss_history, callback, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch,
                    anchors, bbox_util, roi_helper)
            lr = lr*0.96
            K.set_value(model_rpn.optimizer.lr, lr)
            K.set_value(model_all.optimizer.lr, lr)
