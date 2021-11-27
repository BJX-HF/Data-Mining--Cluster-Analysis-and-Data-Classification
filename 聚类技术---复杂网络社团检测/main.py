#GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
G = nx.read_gml('karate/karate.gml',label='id')
num_n = G.number_of_nodes()
num_e = G.number_of_edges()

S = np.zeros((num_n+1,num_n+1))
for i in G:
    for j in G:
        S[i,j] = len(set(G.adj[i])&set(G.adj[j]))/len(set(G.adj[i])|set(G.adj[j]))

#计算类的密度--平均密度
def ComputeDensityAVG(club):
    if len(club)==1:
        return 1
    sum_density = 0.0
    times = 0
    for i in club:
        for j in club:
            if(i==j): continue
            sum_density += S[i,j]
            times += 1
    return sum_density/times
#计算类的密度--最小密度
def ComputeDensityMIN(club):
    density = float('inf')
    for i in club:
        for j in club:
            if(i==j): continue
            density = min(S[i,j],density)
    return density

def FindClubs(threshold=0.25,compute_density=ComputeDensityAVG):
    clubs = []#存放所获得的社团
    club = []#存放当前的社团
    candidate = list(G.nodes)#存放剩余节点，初始为全部节点
    while(len(candidate)>0):
        if(club==[]):#当前社团没有节点，则任选一个节点放入
            firstOne = candidate[random.randint(0,len(candidate)-1)]
            club.append(firstOne)
            candidate.remove(firstOne)
            club_density = compute_density(club)#类密度
            if(len(candidate)==0):
                clubs.append(club)#如果没有节点了，将当前社团加入clubs
        else:#当前社团有节点，利用贪心算法选择加入的节点
            max_density = 0#选择能使加入后的社团密度最大的点，初始化当前密度为0
            best_one = 0
            for i in candidate:
                new_club = club+[i,]
                max_density,best_one = (compute_density(new_club),i) if(compute_density(new_club)>max_density) else (max_density,best_one)
            if(max_density>=threshold):#最优密度大于阈值，该节点加入社团
                club.append(best_one)
                candidate.remove(best_one)
                club_density = max_density
                if(len(candidate)==0):#如果没有节点了，将当前社团加入clubs
                    clubs.append(club)
            else:#当前状况下，最优的密度也低于阈值，则放弃加入节点，当前社团加入clubs
                clubs.append(club)
                club = []
    return clubs


def GetRandomColor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return '#'+color


threshold = float(input("请输入阈值"))
position = nx.spring_layout(G)
clubs = FindClubs(threshold,ComputeDensityAVG)
for club in clubs:
    # nx.draw_networkx_nodes(G, pos=position, nodelist=club, node_color=GetRandomColor(),label='threshold=0.25')
    nx.draw(G, pos=position, nodelist=club, node_color=GetRandomColor(), with_labels=True)
plt.title("threshold=")
plt.show()