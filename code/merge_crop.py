import pandas as pd 
import os
import numpy as np

for fold in range(5):
    for phase in ["train", "val"]:
        df = pd.read_csv("{}_fold{}.csv".format(phase, fold))
        Id_list = []
        Subject_Focus_list = []
        Eyes_list = []
        Face_list = []
        Near_list = []
        Action_list = []
        Accessory_list = []
        Group_list = []
        Collage_list = []
        Human_list = []
        Occlusion_list = []
        Info_list = []
        Blur_list = []
        Pawpularity_list = []

        for index, row in df.iterrows():
            Id = row['Id']
            path = os.path.join("crop", Id + ".jpg")
            Id_list.append(Id)
            Subject_Focus_list.append(row["Subject Focus"])
            Eyes_list.append(row["Eyes"])
            Face_list.append(row["Face"])
            Near_list.append(row["Near"])
            Action_list.append(row["Action"])
            Accessory_list.append(row["Accessory"])
            Group_list.append(row["Group"])
            Collage_list.append(row["Collage"])
            Human_list.append(row["Human"])
            Occlusion_list.append(row["Occlusion"])
            Info_list.append(row["Info"])
            Blur_list.append(row["Blur"])
            Pawpularity_list.append(row["Pawpularity"])
            if os.path.exists(path):
                Id_list.append(Id + "_crop")
                Subject_Focus_list.append(row["Subject Focus"])
                Eyes_list.append(row["Eyes"])
                Face_list.append(row["Face"])
                Near_list.append(row["Near"])
                Action_list.append(row["Action"])
                Accessory_list.append(row["Accessory"])
                Group_list.append(row["Group"])
                Collage_list.append(row["Collage"])
                Human_list.append(row["Human"])
                Occlusion_list.append(row["Occlusion"])
                Info_list.append(row["Info"])
                Blur_list.append(row["Blur"])
                Pawpularity_list.append(row["Pawpularity"])
        new_df = pd.DataFrame(list(zip(Id_list, Subject_Focus_list, Eyes_list, Face_list, Near_list, Action_list, Accessory_list, Group_list, Collage_list, Human_list, Occlusion_list, Info_list, Blur_list, Pawpularity_list)), columns=["Id", "Subject Focus", "Eyes", "Face","Near","Action","Accessory","Group","Collage","Human","Occlusion","Info","Blur", "Pawpularity"])
        new_df.to_csv('{}_fold_crop{}.csv'.format(phase, fold), index=False)

    

