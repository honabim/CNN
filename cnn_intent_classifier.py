from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import typing
from builtins import zip
import os
import io
from future.utils import PY3
from typing import Any, Optional
from typing import Dict
from typing import List
from typing import Text
from typing import Tuple

import numpy as np

from rasa_nlu import utils
from rasa_nlu.classifiers import INTENT_RANKING_LENGTH
from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

import os
import sys
logger = logging.getLogger(__name__)

class CnnIntentClassifier(Component):
    """Intent classifier using the CNN"""

    name="cnn_intent_classifer"
    provides = ["intent", "intent_ranking"]
    def __init__(self, clf=None, le=None):
        self.clf=clf

    def transform_labels_str2num(self, labels):
        # type: (List[Text]) -> np.ndarray
        """Transforms a list of strings into numeric label representation.

        :param labels: List of labels to convert to numeric representation"""

        return self.le.fit_transform(labels)

    def train(self,training_data,config,**kwargs):
        labels = [e.get("intent") for e in training_data.intent_examples]
        if len(set(labels)) < 2:
            logger.warn("Can not train an intent classifier. Need at least 2 different classes. " +
                        "Skipping training of intent classifier.")
        else:
            #Check the existing of D      
            if not (os.path.isdir('data_cnn')):
                os.mkdir('data_cnn')
    
            for label in labels:
                l='data_cnn/Data.'+label
                f=open(l,'w+')
                f.write('')
                f.close()
            count=0
            #print(training_data.training_examples)
            print("len=",len(training_data.intent_examples))
            #print(training_data.intent_examples[50].get("intent"))
            #----------------------------------
            # p1=training_data.examples_per_intent()
            # print("p1=",p1)
            # p2=training_data.sorted_intent_examples()
            # print("p2=",p2)
            print(training_data.intent_examples)
            for example in training_data.intent_examples:
                count=count+1
                print("------------------------")
                print("example=",example)
                print("------------------------")
                if(example.get("intent")=="findRestaurantsByCity"):
                    f1=open('data_cnn/Data.findRestaurantsByCity','a+',encoding="utf8", errors='ignore')
                    f1.write(example.text+'.'+'\n')
                    f1.close()
                elif(example.get("intent")=="greet"):
                    f2=open('data_cnn/Data.greet','a+',encoding="utf8", errors='ignore')
                    f2.write(example.text+'.'+'\n')
                    f2.close()
                elif(example.get("intent")=="bye"):
                    f3=open('data_cnn/Data.bye','a+',encoding="utf8", errors='ignore')
                    f3.write(example.text+'.'+'\n')
                    f3.close()
                elif(example.get("intent")=="affirmative"):
                    f4=open('data_cnn/Data.affirmative','a+',encoding="utf8", errors='ignore')
                    f4.write(example.text+'.'+'\n')
                    f4.close()
                elif(example.get("intent")=="negative"):
                    f5=open('data_cnn/Data.negative','a+',encoding="utf8", errors='ignore')
                    f5.write(example.text+'.'+'\n')
                    f5.close()

                # if(example.get("intent")=="findRestaurantsByCity"):
                #     f1=open('data_cnn/Data.findRestaurantsByCity','a+',encoding="utf8", errors='ignore')
                #     f1.write(example.text+'.'+'\n')
                #     f1.close()
                # elif(example.get("intent")=="greet"):

                # elif(example.get("intent")=="bye"):
                # elif(example.get("intent")=="affirmative"):
                # elif(example.get("intent")=="negative"):
                
            os.system('python train_cnn.py')
    def persist(self,model_dir):
        return {
            "cnn_classifer_sklearn":"run_cnn"
        }
    def process(self,content, message, **kwargs):
        print(" in cnn model, content=",content)
        if not (os.path.isdir('data_evaluate_cnn')):
            os.mkdir('data_evaluate_cnn')
        f6=open('data_evaluate_cnn/Data.findRestaurantsByCity','w+')
        f6.write(content+'.')
        f6.close()

        f7=open('data_evaluate_cnn/Data.greet','w+')
        f7.write(content+'.')
        f7.close()

        f8=open('data_evaluate_cnn/Data.bye','w+')
        f8.write(content+'.')
        f8.close()

        f9=open('data_evaluate_cnn/Data.affirmative','w+')
        f9.write(content+'.')
        f9.close()

        f10=open('data_evaluate_cnn/Data.negative','w+')
        f10.write(content+'.')
        f10.close()
       
        os.system('python eval.py --eval_train --checkpoint_dir="./run_cnn/model/checkpoints/')
        f11=open("run_cnn\model\prediction.txt","r")
        f12=f11.readlines()
        label_dic={'greet':'4.0','findRestaurantsByCity':'3.0','bye':'2.0','affirmative':'1.0','negative':'0.0'}
        temp=0
        # for x in label_dic: 
        #     dem=0
        #     for y in f12:
        #         if(len(y)!=1):
        #             s=y[len(y)-4]+y[len(y)-3]+y[len(y)-2]
        #             if(s==label_dic[x]):
        #                 dem=dem+1
        #         else:
        #             continue
        #     print("name: ",x,"confidence: ",dem/5)
        #     if(dem>temp):
        #         temp=dem
        #         result_str=x

        import numpy as np
        f13=open("run_cnn\model\scores.txt","r")
        scores_str=(f13.read().split('\n')[0].split(' '))
        scores_str.pop(5)
        #print("scores_str=",len(scores_str),type(scores_str))
        #scores_str=[16,-123,1,10,100]
        scores_nu=[]
        for score in scores_str:
            scores_nu.append(float(score))
        #softmax
        print("----------------------------------")
        e_Z = np.exp(scores_nu)
        scores_nu = e_Z / e_Z.sum(axis = 0)
        print("scores_nu=",scores_nu)
        score_dic=[]
        for i in range(0,5):
            if(i==0):
                result_str="negative"
            elif(i==1):
                result_str="affirmative"
            elif(i==2):
                result_str="bye"
            elif(i==3):
                result_str="findRestaurantsByCity"
            elif(i==4):
                result_str="greet"
            score_dic.append({'confidence':scores_nu[i],'key':i,'intent':result_str})
        #print(score_dic)
        import operator
        score_dic.sort(key=operator.itemgetter('confidence'), reverse=True)
        print(score_dic)
        print("----------------------------------")
        # index_max=np.argmax(scores_nu)
        # if(index_max==0):
        #     result_str="negative"
        # elif(index_max==1):
        #     result_str="affirmative"
        # elif(index_max==2):
        #     result_str="bye"
        # elif(index_max==3):
        #     result_str="findRestaurantsByCity"
        # elif(index_max==4):
        #     result_str="greet"
        # result_score=scores_nu[index_max]

    
        intent={"name":score_dic[0]['intent'],"confidence":score_dic[0]['confidence'] }
        intent_ranking=[{'name': score_dic[0]['intent'], 'confidence':score_dic[0]['confidence']}, {'name': score_dic[1]['intent'], 'confidence': score_dic[1]['confidence']}, {'name': score_dic[2]['intent'], 'confidence': score_dic[2]['confidence']}, {'name':score_dic[3]['intent'], 'confidence': score_dic[3]['confidence']}, {'name':score_dic[4]['intent'], 'confidence': score_dic[4]['confidence']}]
        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)
        print("in cnn, message=",message)
        # f13=open('ketqua.txt','a+')
        # f13.write('bye')
        # f13.close()

            






    
                
