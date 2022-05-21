#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAME: Nithilasaravanan Kuppan
ID: 260905444
MAIL: nithilasaravana.kuppan@mail.mcgill.ca

ECSE 683 - ASSIGNMENT 2 - Reproduce (atleast) 3 characters using Imitation Learning methods
Method Used: Regression using Random Forest
Word Written: GROWL
"""

#For loading data from MAT files
from scipy.io import loadmat
import numpy as np

#For running regression models
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#For robot simulations
import pybullet as p
import time
import math
import pybullet_data

#For Kinematic check
import sys


#Function to load data from MAT files
def loadData(letter):
    data_pos = []
    data_vel = []
    location = 'data/2Dletters/'+letter.upper()+'.mat'
    data = loadmat(location)
    data_pos = [d['pos'][0][0].T for d in data['demos'][0]]
    data_vel = [d['vel'][0][0].T for d in data['demos'][0]]
    
    return data_pos, data_vel


#Function to split data into training and testing
def splitDataForModel(data_pos, data_vel):
    #Train data
    pos_train_set = data_pos[0:(len(data_pos) - 1)]
    vel_train_set = data_vel[0:(len(data_vel) - 1)]
    
    pos_train = np.array([item for items in pos_train_set for item in items])
    vel_train = np.array([item for items in vel_train_set for item in items])
    
    #Test data
    pos_test = data_pos[-1]
    vel_test = data_vel[-1]
    
    return pos_train, vel_train, pos_test, vel_test


#Function to run the regression model and get the velocities    
def regressionModel(pos_train, vel_train, pos_test, vel_test, letter):
    rfr = RandomForestRegressor(n_estimators=1000,criterion = 'mse', oob_score=True, random_state = 47, n_jobs = -1)
    rfr.fit(pos_train, vel_train)
    
    #Train Metrics
    print('\n')
    pred_train= rfr.predict(pos_train)
    print(f'Train MSE for {letter} -> {round(np.sqrt(mean_squared_error(vel_train,pred_train)),2)}')
    print(f'Train R2 for {letter} - > {round(r2_score(vel_train, pred_train),2)}')
    
    #Test Metrics
    pred_test= rfr.predict(pos_test)
    print(f'Test MSE for {letter} -> {round(np.sqrt(mean_squared_error(vel_test,pred_test)),2)}') 
    print(f'Test R2 for {letter} -> {round(r2_score(vel_test, pred_test),2)}')
    print('\n')
    
    return pred_test


#Function to add the Z component to the velocity and scale down
def addZ(vel):
    factor = 0.025
    z_vel = np.zeros((len(vel),1))
    vels = np.hstack((vel, z_vel))
    velocity = [[] for idx in range(len(vels))] 
    for i in range(len(vels)): 
        for j in range(len(vels[i])): 
            velocity[i] += [factor * vels[i][j]]
    
    return velocity


#Function to find the current joint positions
def getCurrentJointPos(kukaId):
    allJoints = [0,1,2,3,4,5,6]
    current_js = p.getJointStates(kukaId, allJoints)
    current_jpos = [current_js[i][0] for i in range(len(allJoints))]
    
    return current_jpos


#Function to check if the velocites are in the range
def velocityInRange(vel):
    limit = 10 - 2 # Given limit is 2 but to be safe, keeping it under +/- 8
    for i in range(len(vel)):
        if (vel[i] > -limit and vel[i] < limit):
            print('.', end = '\r')
            return True
        else:
            print('X X Velocity not in Range! X X')
            return False
            
            


#Master Function that takes a letter.mat file, does everything and returns the final cleaned predicted velocity
def getMeVelocity(letter):
    pos_data, vel_data = loadData(letter)
    train_pos, train_vel, test_pos, test_vel = splitDataForModel(pos_data, vel_data)
    pred_vel = regressionModel(train_pos, train_vel, test_pos, test_vel, letter)
    final_vel = addZ(pred_vel)
    
    return final_vel



def main():
    #Getting all the predicted velocities for the letters
    velocity_G = getMeVelocity('G')
    velocity_R = getMeVelocity('R')
    velocity_O = getMeVelocity('O')
    velocity_W = getMeVelocity('W')
    velocity_L = getMeVelocity('L')
    
    time.sleep(2) #To make sure all the stdout prints are visible
   
    
    #Loading the robot and setting Kuka robot arm up
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    #p.loadURDF("plane.urdf", [0, 0, -0.3]) NOT LOADING PLANE FOR BETTER VISIBILITY
    kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])

    p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
    kukaEndEffectorIndex = 6
    numJoints = p.getNumJoints(kukaId)

    rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    for i in range(numJoints):
        p.resetJointState(kukaId, i, rp[i])

    group = 0 #other objects don't collide with Kuka
    mask = 0 # don't collide with any other object
    p.resetJointState(kukaId, jointIndex= 6, targetValue=-2, targetVelocity=-1)
    p.setCollisionFilterGroupMask(kukaId, 0, group, mask) 
    p.setCollisionFilterGroupMask(kukaId, 1, group, mask)
    p.setCollisionFilterGroupMask(kukaId, 2, group, mask)
    p.setCollisionFilterGroupMask(kukaId, 3, group, mask)
    p.setCollisionFilterGroupMask(kukaId, 4, group, mask)
    p.setCollisionFilterGroupMask(kukaId, 5, group, mask)
    p.setCollisionFilterGroupMask(kukaId, 6, group, mask)
   
    
    prevPose1 = [0, 0, 0]
    hasPrevPose = 0
    trailDuration = 0

    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(0)
    
    """
    Robot Arm Navigation begins here
    """
    
    #Moving the arm to a bit to the leftside to allow room for writing
    startPos = [-0.3,0.7,0.3]
    jointPoses = p.calculateInverseKinematics(kukaId,kukaEndEffectorIndex, startPos, solver = 0)
    for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i],
                                    targetVelocity=0,
                                    force=100,
                                    positionGain=0.03,
                                    velocityGain=1)
    for i in range(80):
        p.stepSimulation()
        time.sleep(1/50)
        
    #Block for writing the first character - Letter G
    for i in range(len(velocity_G)):
        velocity = [velocity_G[i][0], velocity_G[i][1], velocity_G[i][2]]
        q = getCurrentJointPos(kukaId)
        Jacobian_t, Jacobian_r = p.calculateJacobian(kukaId, 6, [0]*3, q, [0]*7, [0]* 7)
        J_inv = np.linalg.pinv(Jacobian_t)
        jointVel = np.asarray(np.matmul((J_inv), velocity))
        
        check = velocityInRange(jointVel)
        if check == False:
            sys.exit()
        
        for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                        jointIndex=i,
                                        controlMode=p.VELOCITY_CONTROL,
                                        
                                        targetVelocity=jointVel[i])
                                        
        for i in range(1):
            p.stepSimulation()
            time.sleep(1/50) 
        
        ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
        if (hasPrevPose):
            
            p.addUserDebugLine(prevPose1, ls[4], [0, 0, 0], 1, trailDuration)
        prevPose1 = ls[4]
        hasPrevPose = 1

    print('\n Drawn G')

    #Block to move the arm to leave a gap before writing the next letter
    gap_vel = [1.2,-0.8,0]
    gap_q = getCurrentJointPos(kukaId)
    gap_Jacobian_t, gap_Jacobian_r = p.calculateJacobian(kukaId, 6, [0]*3, gap_q, [0]*7, [0]* 7)
    gap_J_inv = np.linalg.pinv(gap_Jacobian_t)
    gap_jointVel = np.asarray(np.matmul(gap_J_inv, gap_vel))
    for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                        jointIndex=i,
                                        controlMode=p.VELOCITY_CONTROL,
                                        
                                        targetVelocity=gap_jointVel[i])
           
    for i in range(25):
        p.stepSimulation()
        time.sleep(1/50) 
    hasPrevPose = 0
   
    #Block for writing the first character - Letter R
    for i in range(len(velocity_R)):
        velocity = [velocity_R[i][0], velocity_R[i][1], velocity_R[i][2]]
        q = getCurrentJointPos(kukaId)
        Jacobian_t, Jacobian_r = p.calculateJacobian(kukaId, 6, [0]*3, q, [0]*7, [0]* 7)
        J_inv = np.linalg.pinv(Jacobian_t)
        jointVel = np.asarray(np.matmul((J_inv), velocity))
        
        check = velocityInRange(jointVel)
        if check == False:
            sys.exit()
    
        for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                        jointIndex=i,
                                        controlMode=p.VELOCITY_CONTROL,
                                        
                                        targetVelocity=jointVel[i])
                                        
        for i in range(1):
            p.stepSimulation()
            time.sleep(1/50) 
        
        ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
        if (hasPrevPose):
            
            p.addUserDebugLine(prevPose1, ls[4], [0, 0, 0], 1, trailDuration)
        prevPose1 = ls[4]
        hasPrevPose = 1
      
        
    print('\n Drawn R')
    
    #Block to move the arm to leave a gap before writing the next letter
    gap_vel = [1.0,1.5,0]
    gap_q = getCurrentJointPos(kukaId)
    gap_Jacobian_t, gap_Jacobian_r = p.calculateJacobian(kukaId, 6, [0]*3, gap_q, [0]*7, [0]* 7)
    gap_J_inv = np.linalg.pinv(gap_Jacobian_t)
    gap_jointVel = np.asarray(np.matmul(gap_J_inv, gap_vel))
    for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                        jointIndex=i,
                                        controlMode=p.VELOCITY_CONTROL,
                                        
                                        targetVelocity=gap_jointVel[i])
           
    for i in range(30):
        p.stepSimulation()
        time.sleep(1/50) 
    hasPrevPose = 0
    
    
    #Block for writing the first character - Letter O
    for i in range(len(velocity_O)):
        velocity = [velocity_O[i][0], velocity_O[i][1], velocity_O[i][2]]
        q = getCurrentJointPos(kukaId)
        Jacobian_t, Jacobian_r = p.calculateJacobian(kukaId, 6, [0]*3, q, [0]*7, [0]* 7)
        J_inv = np.linalg.pinv(Jacobian_t)
        jointVel = np.asarray(np.matmul((J_inv), velocity))
        
        check = velocityInRange(jointVel)
        if check == False:
            sys.exit()
    
        for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                        jointIndex=i,
                                        controlMode=p.VELOCITY_CONTROL,
                                        
                                        targetVelocity=jointVel[i])
                                        
        for i in range(1):
            p.stepSimulation()
            time.sleep(1/50) 
        
        ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
        if (hasPrevPose):
            
            p.addUserDebugLine(prevPose1, ls[4], [0, 0, 0], 1, trailDuration)
        prevPose1 = ls[4]
        hasPrevPose = 1
   
    
    print('\n Drawn O')
    
    #Block to move the arm to leave a gap before writing the next letter
    gap_vel = [1.0,0,0]
    gap_q = getCurrentJointPos(kukaId)
    gap_Jacobian_t, gap_Jacobian_r = p.calculateJacobian(kukaId, 6, [0]*3, gap_q, [0]*7, [0]* 7)
    gap_J_inv = np.linalg.pinv(gap_Jacobian_t)
    gap_jointVel = np.asarray(np.matmul(gap_J_inv, gap_vel))
    for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                        jointIndex=i,
                                        controlMode=p.VELOCITY_CONTROL,
                                        
                                        targetVelocity=gap_jointVel[i])
           
    for i in range(25):
        p.stepSimulation()
        time.sleep(1/50) 
    hasPrevPose = 0
   
    
    #Block for writing the first character - Letter W
    for i in range(len(velocity_W)):
        velocity = [velocity_W[i][0], velocity_W[i][1], velocity_W[i][2]]
        q = getCurrentJointPos(kukaId)
        Jacobian_t, Jacobian_r = p.calculateJacobian(kukaId, 6, [0]*3, q, [0]*7, [0]* 7)
        J_inv = np.linalg.pinv(Jacobian_t)
        jointVel = np.asarray(np.matmul((J_inv), velocity))
        
        check = velocityInRange(jointVel)
        if check == False:
            sys.exit()
    
        for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                        jointIndex=i,
                                        controlMode=p.VELOCITY_CONTROL,
                                        
                                        targetVelocity=jointVel[i])
                                        
        for i in range(1):
            p.stepSimulation()
            time.sleep(1/50) 
        
        ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
        if (hasPrevPose):
            
            p.addUserDebugLine(prevPose1, ls[4], [0, 0, 0], 1, trailDuration)
        prevPose1 = ls[4]
        hasPrevPose = 1
     
    print('\n Drawn W')
        
    #Block to move the arm to leave a gap before writing the next letter
    gap_vel = [0.7,0.5,0]
    gap_q = getCurrentJointPos(kukaId)
    gap_Jacobian_t, gap_Jacobian_r = p.calculateJacobian(kukaId, 6, [0]*3, gap_q, [0]*7, [0]* 7)
    gap_J_inv = np.linalg.pinv(gap_Jacobian_t)
    gap_jointVel = np.asarray(np.matmul(gap_J_inv, gap_vel))
    for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                        jointIndex=i,
                                        controlMode=p.VELOCITY_CONTROL,
                                        
                                        targetVelocity=gap_jointVel[i])
           
    for i in range(25):
        p.stepSimulation()
        time.sleep(1/50) 
    hasPrevPose = 0        
        
        
    #Block for writing the first character - Letter L
    for i in range(len(velocity_L)):
        velocity = [velocity_L[i][0], velocity_L[i][1], velocity_L[i][2]]
        q = getCurrentJointPos(kukaId)
        Jacobian_t, Jacobian_r = p.calculateJacobian(kukaId, 6, [0]*3, q, [0]*7, [0]* 7)
        J_inv = np.linalg.pinv(Jacobian_t)
        jointVel = np.asarray(np.matmul((J_inv), velocity))
        
        check = velocityInRange(jointVel)
        if check == False:
            sys.exit()
    
        for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                        jointIndex=i,
                                        controlMode=p.VELOCITY_CONTROL,
                                        
                                        targetVelocity=jointVel[i])
                                        
        for i in range(1):
            p.stepSimulation()
            time.sleep(1/50) 
        
        ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
        if (hasPrevPose):
            
            p.addUserDebugLine(prevPose1, ls[4], [0, 0, 0], 1, trailDuration)
        prevPose1 = ls[4]
        hasPrevPose = 1   
        
    print('\n Drawn L \n')
        
    #Block to move the arm for the user to clearly see the written word! 
    gap_vel = [1.2,-1.5,0]
    gap_q = getCurrentJointPos(kukaId)
    gap_Jacobian_t, gap_Jacobian_r = p.calculateJacobian(kukaId, 6, [0]*3, gap_q, [0]*7, [0]* 7)
    gap_J_inv = np.linalg.pinv(gap_Jacobian_t)
    gap_jointVel = np.asarray(np.matmul(gap_J_inv, gap_vel))
    for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId,
                                        jointIndex=i,
                                        controlMode=p.VELOCITY_CONTROL,
                                        
                                        targetVelocity=gap_jointVel[i])
           
    for i in range(80):
        p.stepSimulation()
        time.sleep(1/50) 
    hasPrevPose = 0
   
    p.disconnect()
   


if __name__ == '__main__':
    main()