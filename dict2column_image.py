import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 定义三个JSON文件的文件名
json_files = ['output/genre_count.json', 'output/artist_count.json', 'output/style_count.json']

# 初始化一个颜色列表，每个类别分配一个颜色
colors = ['cadetblue', 'sandybrown', 'salmon']

# 初始化一个标签列表，用于图例
class_labels = ['genre', 'artist', 'style']

# 设置全局字体属性，将刻度值的字体更改为serif
plt.rc('font', family='serif')
# 设置全局字体属性，将刻度值和图例的字体更改为加粗
plt.rc('font', weight='bold')

# 创建一个图形
fig, ax = plt.subplots(figsize=(180, 36))


# 循环处理三个JSON文件
for i, json_file in enumerate(json_files):
    with open(json_file, 'r') as file:
        data = json.load(file)

    # 分离属性名和属性值
    labels = list(data.keys())
    values = list(data.values())

    # 创建柱状图，指定颜色和标签
    # plt.bar(labels, values, color=colors[i],width=0.6,align='edge')

    rate_list=[round(num/7.,2) for num in values]
    plt.bar(labels, rate_list,color=colors[i], width=0.6, align='edge')

    # 在每根柱子的顶部显示数值大小,图片比例，小数点后两位
    for label, value in zip(labels, rate_list):
        plt.text(label, value, str(value)+'%', ha='left', va='bottom', fontsize=31, fontweight='bold')

# 添加标题和标签
plt.title('Our Data Distribution',fontsize=200,weight='bold')
# plt.xlabel('class',fontsize=14)
# plt.ylabel('count',fontsize=20)


# 设置横坐标和纵坐标的刻度标签字体大小,调整属性名的显示角度为45度，并减小字体大小
plt.xticks(rotation=60, fontsize=34,weight='bold')
# 放大纵坐标刻度值字体大小
ax.yaxis.set_tick_params(labelsize=80)

# 添加图例
plt.legend(class_labels,fontsize=90)

# 去掉图表的外框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 增加柱与柱之间的间距
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)

# 在指定纵坐标刻度处绘制虚线
# specified_ticks = [0, 1, 2, 3, 4, 5, 6, 7]
# for tick in specified_ticks:
#     plt.axhline(y=tick, color='gray', linestyle='-', linewidth=4.2)

# 格式化纵坐标刻度标签，添加百分号
def format_percent(x, _):
    return f'{x}%'

ax.yaxis.set_major_formatter(FuncFormatter(format_percent))

# 提高图像分辨率（DPI设置为300，可以根据需要调整）
plt.savefig('bar_chart.png')

# 显示柱状图
# plt.show()
