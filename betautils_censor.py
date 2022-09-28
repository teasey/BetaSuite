import cv2
import math
import numpy as np
import random # added
import glob # added
import os

import betautils_config as bu_config

import betaconst
import betaconfig

def censor_scale_for_image_box( image, feature_w, feature_h ):
    if betaconfig.censor_scale_strategy == 'none':
        return(1)
    if betaconfig.censor_scale_strategy == 'feature':
        return( min( feature_w, feature_h ) / 100 )
    if betaconfig.censor_scale_strategy == 'image':
        (img_h,img_w,_) = image.shape
        return( max( img_h, img_w ) / 1000 )

def pixelate_image( image, x, y, w, h, factor, box ): # factor 10 means 100x100 area becomes 10x10
    factor *= censor_scale_for_image_box( image, w, h )
    new_w = math.ceil(w/factor)
    new_h = math.ceil(h/factor)
    image[y:y+h,x:x+w] = cv2.resize( cv2.resize( image[y:y+h,x:x+w], (new_w, new_h), interpolation = cv2.BORDER_DEFAULT ), (w,h), interpolation = cv2.INTER_NEAREST ) 
    return( image )

def blur_image( image, x, y, w, h, factor, box ):
    factor = 2*math.ceil( factor * censor_scale_for_image_box( image, w, h )/2 ) + 1
    image[y:y+h,x:x+w] = cv2.blur( image[y:y+h,x:x+w], (factor, factor), cv2.BORDER_DEFAULT )
    #image[y:y+h,x:x+w] = cv2.GaussianBlur( image[y:y+h,x:x+w], (factor, factor), 0 )
    return( image )

def bar_image( image, x, y, w, h, color, box ):
    color = tuple( reversed( color ) )
    image = np.ascontiguousarray( image )
    image = cv2.rectangle( image, (x,y), (x+w,y+h), color, cv2.FILLED )
    return( image )

def debug_image( image, box ):
    x = box['x']
    y = box['y']
    w = box['w']
    h = box['h']
    color = tuple( reversed( box['censor_style'][1] ) )
    image = np.ascontiguousarray( image )
    image = cv2.rectangle( image, (x,y), (x+w,y+h), color, 3 )
    image = cv2.putText( image, '(%d,%d)'%(x,y),     (x+10,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 )
    image = cv2.putText( image, '(%d,%d)'%(x+w,y+h), (x+10,y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 )
    image = cv2.putText( image, box['label'],        (x+10,y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 )
    image = cv2.putText( image, '%.2f %.1f %.1f'%(box['score'],box['start'],box['end'] ), (x+10, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 )
    return( image )

def censor_image( image, box ):
    if 'blur' == box['censor_style'][0]:
        return( blur_image( image, box['x'], box['y'], box['w'], box['h'], box['censor_style'][1], box ) )
    if 'pixel' == box['censor_style'][0]:
        return( pixelate_image( image, box['x'], box['y'], box['w'], box['h'], box['censor_style'][1], box ) )
    if 'bar' == box['censor_style'][0]:
        return( bar_image( image, box['x'], box['y'], box['w'], box['h'], box['censor_style'][1], box ) )
    if 'debug' == box['censor_style'][0]:
        return( debug_image( image, box ) )

def watermark_image( image ):
    if betaconfig.enable_betasuite_watermark:
        image = np.ascontiguousarray( image )
        (h,w,_) = image.shape
        scale = max( min(w/750,h/750), 1 )
        return( cv2.putText( image, 'Censored with betasuite.net', (20,math.ceil(20*scale)), cv2.FONT_HERSHEY_PLAIN, scale, (0,0,255), math.floor(scale) ) )
    else:
        return( image )

def annotate_image_shape( image ):
    image = np.ascontiguousarray( image )
    return( cv2.putText( image, str( image.shape ), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2 ) )

def process_raw_box( raw, vid_w, vid_h ):
    parts_to_blur = bu_config.get_parts_to_blur()
    label = betaconst.classes[int(raw['class_id'])][0]
    if label in parts_to_blur and raw['score'] > parts_to_blur[label]['min_prob']:
        x_area_safety = parts_to_blur[label]['width_area_safety']
        y_area_safety = parts_to_blur[label]['height_area_safety']
        time_safety = parts_to_blur[label]['time_safety']
        safe_x = math.floor( max( 0, raw['x'] - raw['w']*x_area_safety/2 ) )
        safe_y = math.floor( max( 0, raw['y'] - raw['h']*y_area_safety/2 ) )
        safe_w = math.ceil( min( vid_w-safe_x, raw['w']*(1+x_area_safety) ) )
        safe_h = math.ceil( min( vid_h-safe_y, raw['h']*(1+y_area_safety) ) )
        return( {
            "start": max( raw['t']-time_safety/2,0 ),
            "end":   raw['t']+time_safety/2,
            "x": safe_x, 
            "y": safe_y, 
            "w": safe_w, 
            "h": safe_h ,
            'censor_style': parts_to_blur[label]['censor_style'],
            'label': label,
            'score': raw['score'],
        } )
    
def rectangles_intersect( box1, box2 ):
    if box1['x']+box1['w'] < box2['x']:
        return( False )

    if box1['y']+box1['h'] < box2['y']:
        return( False )

    if box1['x'] > box2['x']+box2['w']:
        return( False )

    if box1['y'] > box2['y']+box2['h']:
        return( False )

    return( True )

def censor_style_sort( censor_style ):
    if censor_style[0] == 'blur':
        return( 1 + 1/censor_style[1] )
    if censor_style[0] == 'bar':
        return( 2 + 1/(2+255*3-sum(censor_style[1])))
    if censor_style[0] == 'pixel':
        return( 3 + censor_style[1] )
    if censor_style[0] == 'debug':
        return( 99 )
            
def collapse_boxes_for_style( piece ):
    style = piece[0]['censor_style'][0]
    strategy = betaconfig.censor_overlap_strategy[style]
    if strategy == 'none':
        return( piece )
    
    if strategy == 'single-pass':
        segments = []
        for box in piece:
            found = False
            for i,segment in enumerate(segments):
                if rectangles_intersect( box, segment ):
                    x = min( segments[i]['x'], box['x'] )
                    y = min( segments[i]['y'], box['y'] )
                    w = max( segments[i]['x']+segments[i]['w'], box['x']+box['w'] ) - x
                    h = max( segments[i]['y']+segments[i]['h'], box['y']+box['h'] ) - y
                    segments[i]['x']=x
                    segments[i]['y']=y
                    segments[i]['w']=w
                    segments[i]['h']=h
                    found = True
                    break
            if not found:
                segments.append( box )

    return( segments )

def randomize_censor_styles(boxes): #added
    for i in range(len(boxes)):
        if boxes[i]['censor_style'][0] == 'random':
            boxes[i]['censor_style'] = randomize_censor_style() 
            for j in range(len(boxes)):
                if boxes[i]['censor_style'] != boxes[j]['censor_style']:
                    labelstring_i = str(boxes[i]['label']).split('_')  
                    labelstring_j = str(boxes[j]['label']).split('_') 
                    if labelstring_i[1] == labelstring_j[1]:
                        boxes[j]['censor_style'] = boxes[i]['censor_style']
    return (boxes)

def randomize_censor_style(): #added
    censor_style_rnd = []
    perc_range = []
    total = 0
    rnd_censor_var = random.uniform(0,1)
    if betaconfig.default_censor_style[0] == 'random':
        random_censor_styles_loc = betaconfig.default_censor_style[1]
    rand_sum = sum(random_censor_styles_loc[0:len(random_censor_styles_loc):2])
    for n in range(0,len(random_censor_styles_loc),2):
        weight = int(random_censor_styles_loc[n]) / int(rand_sum)
        total = total + weight
        perc_range.append(total)
    if len(random_censor_styles_loc) % 2 != 0:
        print('Error: your random_censor_styles seems to miss something: Check the values after default_censor_style in betaconfig.py')
    chosen_random_censor_style = list(map(lambda i: i> rnd_censor_var, perc_range)).index(True)
    censor_style_rnd = random_censor_styles_loc[2*chosen_random_censor_style+1]     
    return (censor_style_rnd)

def compare_boxes(live_boxes, vid_w, vid_h): # added
    for i in range(len(live_boxes)):
        if live_boxes[i]['censor_style'][0] == 'random':
            for j in range(len(live_boxes)):
                if i != j and live_boxes[i]['label'] == live_boxes[j]['label']:
                    diff_x = abs(live_boxes[i]['x'] - live_boxes[j]['x'])
                    diff_y = abs(live_boxes[i]['y'] - live_boxes[j]['y'])
                    diff_w = abs(live_boxes[i]['w'] - live_boxes[j]['w'])
                    diff_h = abs(live_boxes[i]['h'] - live_boxes[j]['h'])
                    if diff_x <= betaconfig.feature_tracking_tolerance * vid_w and diff_y <= betaconfig.feature_tracking_tolerance * vid_h and diff_w <= betaconfig.feature_tracking_tolerance * vid_w and diff_h <= betaconfig.feature_tracking_tolerance * vid_h:
                        if live_boxes[i]['censor_style'][0] == 'random' and live_boxes[i]['censor_style'] != live_boxes[j]['censor_style']:
                            live_boxes[i]['censor_style'] = live_boxes[j]['censor_style']
    return (live_boxes)

def censor_img_for_boxes( image, boxes ):
    try:
        boxes.sort( key=lambda x: ( x['label'], censor_style_sort(x['censor_style']) ) )
    except TypeError:
        print('TypeError during sorting of boxes')
        
    pieces = []
    for box in boxes:
        if len( pieces ) and pieces[-1][0]['label'] == box['label'] and pieces[-1][0]['censor_style']==box['censor_style']:
            pieces[-1].append(box)
        else:
            pieces.append([box])
            
    if betaconfig.default_censor_style[0] == 'random':
        boxes = randomize_censor_styles(boxes) # added
    
    for piece in pieces:
        collapsed_boxes = collapse_boxes_for_style( piece )

        if betaconfig.sticker == True:                          #added
            collapsed_boxes = sticker_tracking(collapsed_boxes) #added
            
        for collapsed_box in collapsed_boxes:
            image = censor_image( image, collapsed_box )
            if betaconfig.borders == True and (collapsed_box['label'] in betaconfig.border_items):
                cb_x = collapsed_box['x']
                cb_y = collapsed_box['y']
                cb_w = collapsed_box['w']
                cb_h = collapsed_box['h']
                for labels in betaconst.classes:
                    if labels[0] == collapsed_box['label']:
                        bordercolor = labels[1]
                image = np.ascontiguousarray( image )
                image = cv2.rectangle( image, (cb_x,cb_y), (cb_x+cb_w,cb_y+cb_h), bordercolor, 3 )
            if betaconfig.sticker == True and (not collapsed_box['label'] in betaconfig.sticker_exceptions):
                image = add_sticker_image( image, collapsed_box, collapsed_box['censor_sticker'] )
                #print('applied sticker: ', collapsed_box['censor_sticker'], ' to box with label ', collapsed_box['label'])

    image = watermark_image( image )

    return( image )

def add_sticker_image( image, box, sticker ):  
    sticker_image = cv2.imread(sticker, cv2.IMREAD_UNCHANGED)
    s_h, s_w = sticker_image.shape[:2]
    s_alpha_channel = sticker_image[:, :, 3] / 255
    s_colors = sticker_image[:, :, :3]
    alpha_mask = np.dstack((s_alpha_channel, s_alpha_channel, s_alpha_channel))
    
   # adjust sticker size to fill box according to witdh box['w']
    if s_w >= s_h:
        resize_aspect = 0.9 * box['w'] / s_w
    else:
        resize_aspect = 0.9 * box['w'] / s_h
    new_dimensions = (int(box['w']), int(s_h* resize_aspect))
    resized_sticker = cv2.resize(sticker_image, new_dimensions, interpolation = cv2.INTER_AREA)

    
    # place sticker so that its centered vertically
    sticker_y = int(box['y'] + box['h']/2 - new_dimensions[1]/2)
    sticker_x = box['x']
    #print('error?: ', sticker_y, box['x'], new_dimensions)
    image = overlay_transparent(image, resized_sticker, sticker_x, sticker_y)
    return ( image )

def overlay_transparent(image, sticker, x, y):
    if image is None:
        print('image is None')
        return image
    if image.shape[2] == 1:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    #The following code will use the alpha channels of the overlay image (sticker) to correctly blend it into the background image (image), use x and y to set the top-left corner of the overlay image. 
    image_width = image.shape[1]
    image_height = image.shape[0]

    if x >= image_width or y >= image_height:
        #print('nothing done, sticker_x ', x, ' was bigger than image_width ', image_width, 'or sticker_y ', y, ' was bigger than image_height ', image_height)
        return image

    h, w = sticker.shape[0], sticker.shape[1]

    if x + w > image_width:
        w = image_width - x
        sticker = sticker[:, :w]
    
    if y + h > image_height:
        h = image_height - y
        sticker = sticker[:h, :]

  
    if sticker.shape[2] < 4:
        sticker = np.concatenate(
            [
                sticker,
                np.ones((sticker.shape[0], sticker.shape[1], 1), dtype = sticker.dtype) * 255
            ],
            axis = 2,
        )


    sticker_image = sticker[..., :3]
    mask = sticker[..., 3:] / 255.0
    try:
        image[y:y+h, x:x+w] = (1.0 - mask) * image[y:y+h, x:x+w] + mask * sticker_image
    except ValueError as e:
        #print('loaded image did not have channels dimensions: dim mask ', mask.shape, ' dim image[y:y+h, x:x+w] ', image[y:y+h, x:x+w].shape, ' x ' , x, ' w ' , w, ' y ' , y, ' h ' , h , ' dim sticker_image ', sticker_image.shape, ' Error: ', e)
        pass

    return image

def choose_sticker( box ):
    # check box size: if wide use wide sticker, if ~square chose square sticker
    #print(abs(1-(box['w'] / box['h'])), 'box_w', box['w'], 'box_h', box['h'], 'label', box['label'])
    if (1 - (box['w'] / box['h'])) <= -0.25:
        square_sticker = False
        #print('sq_sticker = False')
    else:
        square_sticker = True
        #print('sq_sticker = True')
        
    # choose sticker folder accoring to box['label']  <=> if there is no folder with the name box['label'] choose all
    sticker_path_all = '..\\stickers\\all\\'
    sticker_path = '..\\stickers\\' + str(box['label']) + '\\'
    if not os.path.exists( '..\\stickers\\' + str(box['label']) + '\\' ):
        sticker_path = ''
    if square_sticker:
        sticker_path += 'square\\'
        sticker_path_all += 'square\\'
    else:
        sticker_path += 'wide\\'
        sticker_path_all += 'wide\\'
    sticker_path_all_type = [sticker_path_all + "*.png", sticker_path_all + "*.jpeg", sticker_path_all + "*.jpg"]
    sticker_path_type = [sticker_path + "*.png", sticker_path + "*.jpeg", sticker_path + "*.jpg"]
    stickers = []
    for j in range(len(sticker_path_type)):
        stickers += glob.glob(sticker_path_type[j])
    for j in range(len(sticker_path_all_type)):
        stickers += glob.glob(sticker_path_all_type[j])
    censor_sticker = random.choice(stickers)
    censor_sticker = censor_sticker.replace('/', '\\') 
    #print('chosen sticker: ', censor_sticker)   
    
    return (censor_sticker)
    
def sticker_tracking(boxes):
    for i in range(len(boxes)):
        if not 'censor_sticker' in boxes[i]:
            boxes[i]['censor_sticker'] = choose_sticker( boxes[i] ) 
            #print('assigned censor_sticker of box ', i, ' to ', boxes[i]['censor_sticker'])
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if boxes[i]['censor_style'] == boxes[j]['censor_style'] and boxes[i]['censor_sticker'] != boxes[j]['censor_sticker']:
                boxes[j]['censor_sticker'] = boxes[i]['censor_sticker']
                #print('overwritten: censor_sticker of box ', i, ' changed to ', boxes[j]['censor_sticker'])
            elif boxes[i]['censor_style'] != boxes[j]['censor_style'] and boxes[i]['censor_sticker'] != boxes[j]['censor_sticker']:
                labelstring_i = str(boxes[i]['label']).split('_')  
                labelstring_j = str(boxes[j]['label']).split('_') 
                if labelstring_i[1] == labelstring_j[1] and boxes[i]['censor_sticker'] != boxes[j]['censor_sticker']:
                    boxes[j]['censor_sticker'] = boxes[i]['censor_sticker']
                    #print('overwritten: censor_sticker of box ', i, ' changed to ', boxes[j]['censor_sticker'])
    return boxes