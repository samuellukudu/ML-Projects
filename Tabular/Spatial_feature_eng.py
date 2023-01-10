# src/Spatial_feature_eng.py
# used to generate features involving location 
# using longitudes and latitudes

# rotate coordindates
def rt_coords(df):
    df['rot_15_x'] = (np.cos(np.radians(15)) * df['Longitude']) + \
                    (np.sin(np.radians(15)) * df['Latitude'])
    
    df['rot_15_y'] = (np.cos(np.radians(15)) * df['Longitude']) - \
                    (np.sin(np.radians(15)) * df['Latitude'])
    
    df['rot_30_x'] = (np.cos(np.radians(30)) * df['Longitude']) + \
                    (np.sin(np.radians(30)) * df['Latitude'])
    
    df['rot_30_y'] = (np.cos(np.radians(30)) * df['Longitude']) - \
                    (np.sin(np.radians(30)) * df['Latitude'])
    
    df['rot_45_x'] = (np.cos(np.radians(45)) * df['Longitude']) + \
                    (np.sin(np.radians(45)) * df['Latitude'])
    
    df['rot_45_y'] = (np.cos(np.radians(45)) * df['Longitude']) - \
                    (np.sin(np.radians(45)) * df['Latitude'])
    
    return df