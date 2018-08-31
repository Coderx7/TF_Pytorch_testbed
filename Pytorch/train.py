#in the name of God


def move_snapshots(source_directory, postfix):
    '''
    Moves snapshots from source directory into a separate folder (snap)
    with a postfix, in order to make it unique and identifiable as to 
    which snapshots, belong to which test
  
    '''
    import shutil
    import os
    from pathlib import Path 
    
    snap_folder = '{0}{1}_{2}'.format(source_directory, 'snaps',postfix)
    #check if snap exists if not create one 
    
    if (not os.path.exists(snap_folder)):
        os.makedirs(snap_folder)
        
    for file_name in os.listdir(source_directory):
        if (file_name.endswith(".pth.tar")) :
            shutil.move('{0}{1}'.format(source_directory, file_name),
                        '{0}{1}{2}'.format(snap_folder, os.sep, file_name))
                
    print('moved to {0}'.format(snap_folder))         


def train_sequence(Root, Solver_Path, Training_Count = 5, Best_Model_Snap_Count = 10, Max_Iter=None, Test_Iter=None,
          Test_Interval=None, Best_Acc_Threshold_Top1 = 0.40, Best_Acc_Threshold_Top5 = 0.40,  
          Is_Resumed = False, Solver_State_Model = None, Snapshot_Interval = 100000,
          Display_Rate = None, Plot_Refresh_Rate = 100,
          Is_Output_Saved = False, Output_Save_Interval = 5000):
    '''
    
    Starts a training sequence. 
    Args: 
        Root:
            Specify the root folder, containing the folders which hold your training prototxt files.
            IMPORTANT : the directory path for the root needs to have the trailing slash(/) at the end!
        Solver_Path:
            The path to your solver.prototxt file, can be a full or relative path to the root
        Training_Count:
            The number of times you wish the training to be repeated
        Best_Model_Snap_Count:
            The number of last best models you wish to keep, if you set this to 3, the script
            will keep the last 3 best models (Top1 and Top5 alike) 
        Best_Acc_Threshold_Top1/Top5:
            The threshold used for saving/keeping Best models. By setting these arguments, 
            you specify when the script start saving models 
            The input range:  0-1 
            example input : 
                Best_Acc_Threshold_Top1 = 0.40
                Best_Acc_Threshold_Top5 = 0.55 
    '''
    import pathlib

    #change cwd to root
    os.chdir(Root)

    real_solver_path = os.path.realpath(Solver_Path)
    source_dir = pathlib.Path(real_solver_path).parent
    training_count = Training_Count + 1
    
    #run training for 3 times, move all snaps to separate folders 
    for i in range(1, training_count ):
        time_span = Train(Solver_Path, Best_Model_Snap_Count, Max_Iter, Test_Iter,
                          Test_Interval, Best_Acc_Threshold_Top1, Best_Acc_Threshold_Top5,  
                          Is_Resumed, Solver_State_Model, Snapshot_Interval, Display_Rate,
                          Plot_Refresh_Rate, Is_Output_Saved, Output_Save_Interval)
        
        move_snapshots('{0}{1}'.format(source_dir, os.sep), '{0}_{1}'.format(str(i),time_span))
    print('Job is done. Tests are finished.')       
    
    

