#Q2
# import copy
# a = [1,[1,2]]
# b = copy.deepcopy(a)
# c = copy.copy(a)
# print(a,id(a))
# print(b,id(b))
# print(c,id(c))
# a[1][1]=0
# print(a,id(a))
# print(b,id(b))
# print(c,id(c))

#Q4
# f = lambda x,y:x if x>y else y
# print(f(7,3))

#Q5
import threading as th

#Q11
# class Foo(object):
#     def bar(self):
#         print("Foo.bar")
# def bar(self):
#     print("Modified bar")
#
# Foo().bar()
#
# Foo.bar = bar
#
# Foo().bar()


#Q16
# import random
# a = [1,7,46,5,36,9,75]
# random.shuffle(a)
# print(a)

#Q17
# a = "_"
#
# print(a.join(['c','b','a']))
#
# print(a.join(['c','b','a']).split('_'))

#Q20
# a = " aa"
# # print(a)
# # print(a.lstrip())


#Q21
# a = "aAcG"
# print(a.lower())
# print(a.upper())
# print(a.lower().islower())
# print(a.upper().isupper())


#Q28
# a = [[1,2],[4,3],5]
# print(5 in a)
# print(1 in a)
# print([1,2] in a)


#Q29
# print('a' is 'b')
# print(True is False)
# print(True is not False)


#Q31
# a = 16
# print(bin(a))
# print(oct(a))
# print(hex(a))

#Q32
# dict = {'a':1,'b':2,'3':'c'}
# print(dict.keys())


#Q33
# (a,b) = (1,2)
# c,d = 3,4
# [e,f] = 5,6
# print(a,b,c,d,e,f)