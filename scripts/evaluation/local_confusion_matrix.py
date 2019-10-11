import pickle
import sys
import distances_detailed
import seaborn as sns
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from matplotlib import pyplot as plt
sys.path.append('/home/rolczynski/projects/DeepSpeech-Keras')
from source.text import Alphabet


def lighten(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


confusion_matrix, inserts, deletes = pickle.load(open('confusion_matrix.bin', 'rb'))

# ALPHABET = Alphabet('../../../DeepSpeech-Keras/tests/models/base/alphabet.txt')
# fig, ax = plt.subplots(figsize=[7, 7])
# labels = list(ALPHABET._label_to_str)  # Copy the list (not change object inplace)
# labels[0], labels[-1] = '$', '_'  # To be visible on the plot
# ax = sns.heatmap(confusion_matrix, vmax=50, square=True, xticklabels=labels, yticklabels=labels,
#                  annot=False, linewidths=.1, ax=ax, cmap="YlGnBu", cbar_kws={"shrink": 0.5})
# ax.set_xlabel('True label', fontweight="bold", fontsize=12)
# ax.set_ylabel('Predicted label', fontweight="bold", fontsize=12)
# ax.set_title('Confusion Matrix', fontweight="bold", fontsize=15)
#
# plt.tight_layout()
# fig.savefig('evaluation-confusion_matrix.png', dpi=600)
# plt.show()

###########################################################
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

tot_inserts = sum(inserts.values())
tot_deletes = sum(deletes.values())
tot_substitute = confusion_matrix.sum()
tot = tot_inserts + tot_deletes + tot_substitute
group_names = [f'Insert {tot_inserts/tot*100:.2f}\%',
               f'Delete {tot_deletes/tot*100:.2f}\%',
               f'Substitute {tot_substitute/tot*100:.2f}\%']
group_size = [tot_inserts, tot_deletes, tot_substitute]

subgroup_size = [tot_inserts - inserts[' '], inserts[' '],
                 deletes[' '], tot_deletes - deletes[' '],
                 tot_substitute]

# Create colors
cmap = plt.get_cmap('YlGnBu')

# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, textprops={'fontsize': 15},
                  colors=[cmap(0.7), cmap(0.5), cmap(0.2)])
plt.setp(mypie, width=0.3, edgecolor='white')


# Second Ring (Inside)
mypie2, _ = ax.pie(subgroup_size, radius=1.3 - 0.3,
                   colors=[cmap(0), lighten(cmap(0.7)), lighten(cmap(0.5)), cmap(0), cmap(0)])
plt.setp(mypie2, edgecolor='white')
plt.margins(0, 0)
# file_name = os.path.join(directory, 'donut.svg')
fig.savefig('evaluation-donut.png', dpi=600)
plt.show()
pass
