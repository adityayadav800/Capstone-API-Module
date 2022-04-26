from Tensorflow import tf, keras
import numpy as np 
import cv2


src_img= cv2.imread('/content/hindi2.png')
copy = src_img.copy()
height = src_img.shape[0]
width = src_img.shape[1]


def showimages(src_img,bin_img,final):
    cv2.imshow("Source Image", src_img)
    cv2.imshow("Binary Image", bin_img)
    cv2.imshow("Threshold Image", final_thr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def line_array(array):
    list_x_upper = []
    list_x_lower = []
    for y in range(5, len(array)-5):
        s_a, s_p = strtline(y, array)
        e_a, e_p = endline(y, array)
        print(str(s_a) + ',' + str(s_p) + ',' + str(e_a) + ',' + str(e_p) + ',' + str(y))
        if s_a>=7 and s_p>=5:
            list_x_upper.append(y)
        # bin_img[y][:] = 255
        if e_a>=5 and e_p>=7:
            list_x_lower.append(y)
            # bin_img[y][:] = 255
    return list_x_upper, list_x_lower

def strtline(y, array):
    count_ahead = 0
    count_prev = 0
    for i in array[y:y+10]:
        if i > 3:
            count_ahead+= 1  
    for i in array[y-10:y]:
        if i == 0:
            count_prev += 1  
    return count_ahead, count_prev

def endline(y, array):
    count_ahead = 0
    count_prev = 0
    for i in array[y:y+10]:
        if i==0:
            count_ahead+= 1  
    for i in array[y-10:y]:
        if i >3:
            count_prev += 1  
    return count_ahead, count_prev

def endline_word(y, array, a):
    count_ahead = 0
    count_prev = 0
    for i in array[y:y+2*a]:
        if i < 2:
            count_ahead+= 1  
    for i in array[y-a:y]:
        if i > 2:
            count_prev += 1  
    return count_prev ,count_ahead

def end_line_array(array, a):
    list_endlines = []
    for y in range(len(array)):
        e_p, e_a = endline_word(y, array, a)
        #print(e_p, e_a)
        if e_a >= int(0.8*a) and e_p >= int(0.7*a):
            list_endlines.append(y)
    return list_endlines

def refine_endword(array):
    refine_list = []
    for y in range(len(array)-1):
        if array[y]+1 < array[y+1]:
            refine_list.append(array[y])
    refine_list.append(array[-1])
    return refine_list


def refine_array(array_upper, array_lower):
    upperlines = []
    lowerlines = []
    for y in range(len(array_upper)-1):
        if array_upper[y] + 5 < array_upper[y+1]:
            upperlines.append(array_upper[y]-10)
    for y in range(len(array_lower)-1):
        if array_lower[y] + 5 < array_lower[y+1]:
            lowerlines.append(array_lower[y]+10)

    upperlines.append(array_upper[-1]-10)
    lowerlines.append(array_lower[-1]+10)
    
    return upperlines, lowerlines

def letter_width(contours):
    letter_width_sum = 0
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            x,y,w,h = cv2.boundingRect(cnt)
            letter_width_sum += w
            count += 1

    return letter_width_sum/count


def end_wrd_dtct(final_local, i, bin_img, mean_lttr_width):
    count_y = np.zeros(shape = width)
    for x in range(width):
        for y in range(final_local[i],final_local[i+1]):
            if bin_img[y][x] == 255:
                count_y[x] += 1
    #end_lines = end_line_array(count_y, int(mean_lttr_width))
    #endlines = refine_endword(end_lines)
    #print(i)
    '''for x in range(len(count_y)):
        if max(count_y[0:x+1]) >= 3 and max(count_y[x:]) >= 3 and (20-np.count_nonzero(count_y[x-10:x+10])) > 6:
            print(x)'''

    contours, hierarchy = cv2.findContours(lines_img[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letter_width_sum = 0
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            letter_width_sum += w
            count += 1
    if count != 0 :
        mean_width = letter_width_sum / count
    else:
        mean_width = 0
    #print(mean_width)
    spaces = []
    line_end = []
    for x in range(len(count_y)):
        number = int(0.5*int(mean_width)) - np.count_nonzero(count_y[x-int(0.25*int(mean_width)):x+int(0.25*int(mean_width))])
        if max(count_y[0:x + 1]) >= 3 and number >= 0.4*int(mean_width):
            spaces.append(x)
        if max(count_y[x:]) <= 2:
            line_end.append(x)
    if line_end!=[]:
      true_line_end = min(line_end) + 10
    else:
      true_line_end=10
    #spaces = refine_endword(spaces)
    #print(spaces)
    #print(true_line_end)
    reti = []
    final_spaces = []
    for j in range(len(spaces)):
        if spaces[j] < true_line_end:
            if spaces[j] == spaces[j-1] + 1:
                reti.append(spaces[j-1])
            elif spaces[j] != spaces[j-1] + 1 and spaces[j-1] == spaces[j-2] +1:
                reti.append(spaces[j-1])
                retiavg = int(sum(reti)/len(reti))
                final_spaces.append(retiavg)
                reti = []
            elif spaces[j] != spaces[j-1] + 1 and spaces[j-1] != spaces[j-2] +1 and spaces[j] != spaces[j+1] -1:
                final_spaces.append(spaces[j])
        elif spaces[j] == true_line_end:
            final_spaces.append(true_line_end)
    #print(final_spaces)
    for x in final_spaces:
        final_thr[final_local[i]:final_local[i+1], x] = 255
    return final_spaces


def letter_seg(lines_img, x_lines, i):
    copy_img = lines_img[i].copy()
    x_linescopy = x_lines[i].copy()
    
    letter_img = []
    letter_k = []
    
    contours, hierarchy = cv2.findContours(copy_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   
    for cnt in contours:
        if cv2.contourArea(cnt) > 5:
            x,y,w,h = cv2.boundingRect(cnt)
            # letter_img.append(lines_img[i][y:y+h, x:x+w])
            letter_k.append((x,y,w,h))

    letter_width_sum = 0
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            letter_width_sum += h
            count += 1

    #mean_height = letter_width_sum/count

    letter = sorted(letter_k, key=lambda student: student[0])
    print(letter)
    for e in range(len(letter)):
      if e<len(letter)-1:
        if abs(letter[e][0] - letter[e+1][0]) <= 2:
          x,y,w,h = letter[e]
          x2,y2,w2,h2 = letter[e+1]
          if h >= h2:
            letter[e] = (x,y2,w,h+h2)
            letter.pop(e+1)
          elif h < h2:
            letter[e+1] = (x2,y,w2,h+h2)
            letter.pop(e)

    for e in range(len(letter)):
        letter_img_tmp = lines_img[i][letter[e][1]-0:letter[e][1]+letter[e][3]+0,letter[e][0]-0:letter[e][0]+letter[e][2]+0]
        letter_img_tmp = cv2.resize(letter_img_tmp, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        width = letter_img_tmp.shape[1]
        height = letter_img_tmp.shape[0]
        count_y = np.zeros(shape=(width))
        for x in range(width):
            for y in range(height):
                if letter_img_tmp[y][x] == 255:
                    count_y[x] = count_y[x] +1
        print(count_y)
        max_list = []
        for z in range(len(count_y)):
            if z>=5 and z<= len(count_y)-6:
                if max(count_y[z-5:z+6]) == count_y[z] and count_y[z] >= 2:
                    max_list.append(z)
            elif z<5:
                if max(count_y[0:z+6]) == count_y[z] and count_y[z] >= 2:
                    max_list.append(z)
            elif z > len(count_y)-6:
                if max(count_y[z-5:len(count_y)-1]) == count_y[z] and count_y[z] >= 2:
                    max_list.append(z)
        print(max_list)
        rem_list = []
        final_max_list = []
        for z in range(len(max_list)):
            if z > 0:
                if max_list[z]-max_list[z-1] <= 3:
                    rem_list.append(z-1)
        for z in range(len(max_list)):
            if z not in rem_list:
                final_max_list.append(max_list[z])
        print(final_max_list)
        if len(final_max_list) <= 1:
            print(False)
        else:
            max_len = len(final_max_list) - 1
            for j in range(max_len):
                list = count_y[final_max_list[j]:final_max_list[j+1]]
                min_list = sorted(list)[:3]
                avg = sum(min_list)/len(min_list)
                print(avg)

    x_linescopy.pop(0)
    word = 1
    letter_index = 0
    for e in range(len(letter)):
        #print(str(letter[e][0]) + ',' + str(letter[e][1]) + ',' + str(letter[e][2]) + ',' + str(letter[e][3]) + ',' + str(e))
        if x_linescopy !=[]:
          if(letter[e][0]<x_linescopy[0]):
              letter_index += 1
              letter_img_tmp = lines_img[i][letter[e][1]-0:letter[e][1]+letter[e][3]+5,letter[e][0]-2:letter[e][0]+letter[e][2]+2]
              try:
                letter_img = cv2.resize(letter_img_tmp, dsize =(28, 28), interpolation = cv2.INTER_AREA)
              except:
                continue
              cv2.imwrite('./segmented_img/img1/'+str(i+1)+'_'+str(word)+'_'+str(letter_index)+'.jpg', 255-letter_img)
          else:
              x_linescopy.pop(0)
              word += 1
              letter_index = 1
              letter_img_tmp = lines_img[i][letter[e][1]-0:letter[e][1]+letter[e][3]+5,letter[e][0]-2:letter[e][0]+letter[e][2]+2]
              try:
                letter_img = cv2.resize(letter_img_tmp, dsize =(28, 28), interpolation = cv2.INTER_AREA)
              except :
                continue
              cv2.imwrite('./segmented_img/img1/'+str(i+1)+'_'+str(word)+'_'+str(letter_index)+'.jpg', 255-letter_img)
