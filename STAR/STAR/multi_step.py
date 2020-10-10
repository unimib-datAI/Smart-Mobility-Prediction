import numpy as np
import os

def multi_step_2D(model, path_model, hyperparams_name, X_test, Y_test, step):
    # model = build_model(external_dim)
    dic_muilt_rmse = {}
    fname_param = os.path.join(path_model, '{}_cont.h5'.format(hyperparams_name))
    model.load_weights(fname_param)
    print(hyperparams_name)
    nb_flow = 2
    len_closeness = int(hyperparams_name.split('.')[0][-1])
    y_pre = []
    X_test_now = X_test

    # inference
    for i in range(1, step + 1):
        y_pre_inference = model.predict(X_test_now)  # 1
        # expand dims [timeslots, flow, height, width] --> [step, timeslots, flow, height, width]
        y_pre_expand_dims = np.expand_dims(y_pre_inference, axis=0)
        # append in all step
        y_pre.append(y_pre_expand_dims)

        X_test_noremove = X_test_now[0][1:]
        X_test_noremove = X_test_noremove.transpose((1, 0, 2, 3))
        X_test_noremove = X_test_noremove[len_closeness * nb_flow:]
        X_test_noremove = X_test_noremove.transpose((1, 0, 2, 3))

        X_test_remove = X_test_now[0].transpose((1, 0, 2, 3))
        X_test_remove = X_test_remove[:len_closeness * nb_flow]

        y_pre_remove = y_pre_inference.transpose((1, 0, 2, 3))

        if len_closeness > 1:
            for j in range(len_closeness * nb_flow, 3, -nb_flow):
                X_test_remove[j - 2:j] = X_test_remove[j - 4:j - 2]  #
            X_test_remove[0:2] = y_pre_remove
        else:
            X_test_remove[0:2] = y_pre_remove

        X_test_remove = X_test_remove.transpose((1, 0, 2, 3))
        X_test_remove = X_test_remove[:-1]

        X_test_next = np.concatenate((X_test_remove, X_test_noremove), axis=1)
        #
        # make training data
        X_test_makeData = []

        X_test_makeData.append(X_test_next)
        X_test_makeData.append(X_test[1][i:])

        X_test_now = X_test_makeData

    # inverse_transform
    for i in range(len(y_pre)):
        y_pre[i] = (y_pre[i] + 1) / 2 * 1292

    Y_test = (Y_test + 1) / 2 * 1292

    for i in range(len(y_pre)):
        rmse = np.sqrt(np.sum((y_pre[i][0] - Y_test[i:]) ** 2) / (
                    Y_test[i:].shape[0] * Y_test.shape[1] * Y_test.shape[2] * Y_test.shape[3]))
        print("RMSE of step%d=%f'" % (i, rmse))
        dic_muilt_rmse[i] = rmse

    return dic_muilt_rmse
