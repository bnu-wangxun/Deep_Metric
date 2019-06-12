# coding=utf-8
from __future__ import absolute_import


class Mazz():
    def __init__(self, m=16, n=16, map_=[[0, 1], [1, 1]]):
        self.route = []
        self.position = [0, 0]
        self.target = [m, n]
        self.map = map_
              
    def get_avaliable_direction(self, position):
        # position 表示当前位置
        # 返回可以走的方向 如[[0, 1]]表示仅可以向下走
        avaliable_direction = []
        if self.map[position[0]+1, position[1]] == 1:
            avaliable_direction.append([1, 0])
        if self.map[position[0], position[0]+1] == 1:
            avaliable_direction.append([0, 1])
        if self.map[position[0], position[0]-1] == 1:
            avaliable_direction.append([0, -1])
        if self.map[position[0]-1, position[1]] == 1:
            avaliable_direction.append([-1, 0])
        return avaliable_direction
    
    def get_route(self, position, target, last_route):
        avaliable_direction = self.get_avaliable_direction(self, position)
        if position == target:
            return []

        for direction in avaliable_direction:
            position_ = [position[0]+direction[0], position[1]+direction[1]]
            if position not in last_route:
                last_route.append(position)
            return [direction]+ self.get_route(self, position_, target, last_route)

    def Solution(self):
        last_route = []
        return self.get_route(self.position, self.target, last_route)




    
    
    
    
        
    
        
