import logging
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import six
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import median_survival_times
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# from ..plot.pyplot import subplots

# plt.rcParams['font.family'] = 'monospace'
# plt.rcParams['font.family'] = 'sans-serif'

def subplots(rows=1, cols=4, w=12, h=4, return_f=False):
    """
    axs = subplots(rows=1, cols=4, w=8)

    sns.boxplot(y=performance.loc[1].scalars, ax=axs[0])
    sns.boxplot(y=performance.loc[7].scalars, ax=axs[1])

    [ax.set_ylim([0.7, 1]) for ax in axs];
    """
    f = plt.figure(constrained_layout=True, figsize=(w, h))
    # gs = f.add_gridspec(rows, cols)
    gs = GridSpec(rows, cols, figure=f)
    axs = []
    for i in range(rows):
        for j in range(cols):
            ax_ = f.add_subplot(gs[i, j])
            ax_.set_facecolor((1, 1, 1, 0))
            axs.append(ax_)

    if return_f:
        return f, axs
    else:
        return axs

def fix_string(s, v, m):
    s = s.split('\t')
    sx = ''
    for ix, i in enumerate(s):
        sx += i + ' '*int(1.*(m[ix] - v[ix])) + ' '
    
    return sx


class KMPlot():
    def __init__(self, data, time, event, label, **kwargs):
        '''
        Example: 

        axs = subplots(cols=1, rows=1, w=6, h=4)
        KMPlot(data, time=time, event=event, label=['bin_risk', 'Treatment']).plot(
            labels = ['GP_{}'.format('IO'), 'GN_{}'.format('IO'), 'GP_{}'.format('Chemo'), 'GN_{}'.format('Chemo')],
            ax=axs[1],
            comparisons=[
                ['GP_{}'.format('IO'), 'GP_{}'.format('Chemo'), 'GP(IO vs Chemo): '], 
                ['GN_{}'.format('IO'), 'GN_{}'.format('Chemo'), 'GN(IO vs Chemo): '], 
                ['GP_{}'.format('IO'), 'GN_{}'.format('IO'), 'IO(GP vs GN): '], 
                ['GP_{}'.format('Chemo'), 'GN_{}'.format('Chemo'), 'Chemo(GP vs GN): ']
            ],
            title='PFS - IO vs Chemo',
        );

        Optional: 
        
        saturation=1.0,
        linewidth=1.5,
        palette='Paired',
        template_color = 'black', xy_font_size = 12,
        hr_color = 'black',
        x_legend = 0.15, y_legend = 0.95, legend_font_size=10,
        label_height_adj=0.055,
        x_hr_legend = 0.0, y_hr_legend = -0.3, hr_font_size=10,
        

        Contact: gaarangoa
        '''
        
        self.colors = {}
        
        self.fit(data, time, event, label, **kwargs)
    
    def compare(self, ):
        pass
    
    def plot(self, labels=None, **kwargs):
        '''
        label[optional]: Label(s) to plot
        linestyle[optional]:  list same dim as labels
        color[optional]:  list same dim as labels
        linewidth[optional]: list same dim as labels
        legend_font_size[optional]: font size for legend
        legend_labelspacing[optional]: 
        saturation[optional]
        label_height_adj: adjust space between labels (y axis)
        xy_font_size: font size of x and y labels
        comparisons: make comparisons between two curves [[tar, ref], [io, soc], [D, D+T]]
        palette: "Paired"
        template_color: '#7a7974'
        adj_label_loc: 0.1
        hr_color: 'black' # Color for hr layer
        display_labels = [comp1, comp2]

        Example: 

        axs = subplots(cols=1, rows=1, w=6, h=4)
        KMPlot(data, time=time, event=event, label=['bin_risk', 'Treatment']).plot(
            labels = ['GP_{}'.format('IO'), 'GN_{}'.format('IO'), 'GP_{}'.format('Chemo'), 'GN_{}'.format('Chemo')],
            ax=axs[1],
            comparisons=[
                ['GP_{}'.format('IO'), 'GP_{}'.format('Chemo'), 'GP(IO vs Chemo): '], 
                ['GN_{}'.format('IO'), 'GN_{}'.format('Chemo'), 'GN(IO vs Chemo): '], 
                ['GP_{}'.format('IO'), 'GN_{}'.format('IO'), 'IO(GP vs GN): '], 
                ['GP_{}'.format('Chemo'), 'GN_{}'.format('Chemo'), 'Chemo(GP vs GN): ']
            ],
            title='PFS - IO vs Chemo',
        );

        Optional: 
        
        saturation=1.0,
        linewidth=1.5,
        palette='Paired',
        template_color = 'black', xy_font_size = 12,
        hr_color = 'black',
        x_legend = 0.15, y_legend = 0.95, legend_font_size=10,
        label_height_adj=0.055,
        x_hr_legend = 0.0, y_hr_legend = -0.3, hr_font_size=10,


        '''

        
        label = labels

        if label == None:
            plot_labels = self.labels
        elif type(label) == list:
            plot_labels = label
        else:
            plot_labels = [label]
        
        display_labels = kwargs.get('display_labels', None)
        ax = kwargs.get('ax', False)
        # if ax == False:
        #     ax = subplots(cols=1, rows=1, w=6, h=4)[0]
        
        colors = kwargs.get('colors', sns.color_palette(kwargs.get('palette', 'Paired'), 100, desat=kwargs.get('saturation', 1)))
        linestyle = kwargs.get('linestyle', ['-']*len(plot_labels))
        xy_font_size = kwargs.get('xy_font_size', 12)
        label_height_adj = kwargs.get('label_height_adj', 0.05)
        template_color = kwargs.get('template_color', 'black')
        to_compare = kwargs.get('comparisons', [])
        title_weight = kwargs.get('title_weight', 'normal')
        title_size = kwargs.get('title_size', 14)
        
        if type(colors) == str:
            colors = [colors]
        
        
        label_max_l = [self.label_names_size['__label__']]
        for lx, label_ in enumerate(plot_labels):
            label_max_l.append(self.label_names_size[label_])
            self.colors[label_] = colors[lx]
            self.kmfs[label_].plot(
                ci_show=kwargs.get('ci_show', False), 
                show_censors=True,
                color = colors[lx],
                linestyle = linestyle[lx],
                linewidth = kwargs.get('linewidth', 1.5),
                ax=ax
            )
            
            # median survival time
            ax.axvline(self.kmfs[label_].median_survival_time_, 0, 0.5, ls = '--', color = self.colors[label_], lw = 1)
            ax.plot((0, self.kmfs[label_].median_survival_time_), (0.5, 0.5),  ls = '--', color = '#a19595', lw = 1)
            sns.scatterplot(x=[self.kmfs[label_].median_survival_time_], y=[0.5], ax=ax, s=50, color='white', edgecolor=self.colors[label_], alpha=1.0)
            
        
        self.colors['__label__'] = 'black'

        # plt.rcParams['font.family'] = kwargs.get('font_family_labels', 'Roboto Mono for Powerline')
        x_legend=kwargs.get('x_legend', 0.15)
        y_legend=kwargs.get('y_legend', 0.95)
        legend_font_size=kwargs.get('legend_font_size', 10)

        label_max_l = np.array(label_max_l).max(axis=0)
        for lx, label_ in enumerate(['__label__'] + plot_labels):                
            vx = fix_string(self.label_names_list[label_], self.label_names_size[label_], label_max_l)

            ax.annotate(
                vx, 
                xy=(x_legend, y_legend -lx*label_height_adj), 
                xycoords='axes fraction', 
                xytext=(x_legend, y_legend -lx*label_height_adj),
                weight='bold', 
                size=legend_font_size, 
                color=self.colors[label_],
                # bbox=dict(fc='white', lw=0, alpha=0.3)
            )

        # ax.annotate(
        #     '', 
        #     xy=(-0.01, y_legend -(lx)*label_height_adj), 
        #     xycoords='axes fraction', 
        #     xytext=(1, y_legend -(lx)*label_height_adj), 
        #     arrowprops=dict(arrowstyle="-", color='k'),
        # )

        # Cox PH Fitters for HR estimation
        xcompare = [[('__label__', '__label__'), "\tHR\t(95% CI)\tP value"]]
        xinfo = [[len(i) for i in "\tHR\t(95% CI)\tP value".split('\t')]]
        for cx, item in enumerate(to_compare):
            
            if len(item) == 3:
                [tar, ref, hr_label] = item
            else: 
                [tar, ref] = item
                hr_label = '{} - {}: '.format(tar, ref)

            x = self.data[self.data.__label__.isin([tar, ref])][[self.time, self.event, '__label__']].copy().reset_index(drop=True)
            x.__label__.replace(ref, 0, inplace=True)
            x.__label__.replace(tar, 1, inplace=True)
            x.__label__ = x.__label__.astype(float)

            cph = CoxPHFitter().fit(x, duration_col = self.time, event_col = self.event) 
            cph = cph.summary[['exp(coef)', 'p', 'exp(coef) lower 95%', 'exp(coef) upper 95%']].reset_index().to_dict()
            cph = {
                "HR": cph.get('exp(coef)').get(0),
                "HR_lo": cph.get('exp(coef) lower 95%').get(0),
                "HR_hi": cph.get('exp(coef) upper 95%').get(0),
                "P": cph.get('p').get(0),
            }
            
            # color for HR 
            hr_color = kwargs.get('hr_color', self.colors[tar])
            
            # xinfo_ = '{}\tHR={:.2f}\t(CI 95%: {:.2f} - {:.2f})\tP-value={:.2e}'.format(hr_label, cph.get('HR'), cph.get('HR_lo'), cph.get('HR_hi'), cph.get('P'))
            xinfo_ = '{}\t{:.2f}\t({:.2f}-{:.2f})\t{:.2e}'.format(hr_label, cph.get('HR'), cph.get('HR_lo'), cph.get('HR_hi'), cph.get('P'))
            xinfo.append([len(i) for i in xinfo_.split('\t')])

            xcompare.append([
                (tar, ref), xinfo_
            ])

        

        x_hr_legend=kwargs.get('x_hr_legend', 0)
        y_hr_legend=kwargs.get('y_hr_legend', -0.3)
        hr_font_size=kwargs.get('hr_font_size', 10)

        max_values = np.array(xinfo)
        for ix, [k, v] in enumerate(xcompare):
            
            tar, ref = k 
            if len(xcompare) == 1: continue
            hr_color = kwargs.get('hr_color', self.colors[tar])
            
            vx = fix_string(v, max_values[ix], max_values.max(axis=0))

            ax.annotate(
                vx, 
                xy=(x_hr_legend, y_hr_legend - ix*label_height_adj), 
                xycoords='axes fraction', 
                xytext=(x_hr_legend, y_hr_legend - ix*label_height_adj),
                weight='bold', 
                size=hr_font_size, 
                color=hr_color,
                # bbox=dict(fc='white', lw=0, alpha=0.5)
            )

        # plt.rcParams['font.family'] = kwargs.get('font_family', '')   
        
        
        ax.set_ylim([-0.03, 1])
        ax.set_ylabel(kwargs.get('ylab', 'Survival Probability'), weight='bold', fontsize=xy_font_size, color=template_color)
        ax.set_xlabel(kwargs.get('xlab', 'Timeline'), weight='bold', fontsize=xy_font_size, color=template_color)
        ax.tick_params(axis='x', colors=template_color)
        ax.tick_params(axis='y', colors=template_color)
        
        ax.xaxis.set_tick_params(labelsize=xy_font_size-1)
        ax.yaxis.set_tick_params(labelsize=xy_font_size-1)
        
        ax.set_title(kwargs.get('title', ''), fontsize=title_size, color=template_color, weight=title_weight)

        ax.set_yticks(ax.get_yticks()[-6:])
        
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(1.0)
            ax.spines[axis].set_color(template_color)

        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0.0)
            
        ax.get_legend().remove()
        # ax.savefig("/projects/aa/mlops/knkf180/code/new_download/ods-bikg/eda/results/km_plot.png")
        # plt.savefig("/projects/aa/mlops/knkf180/code/new_download/ods-bikg/eda/results/km_plot.png")
        
    def fit(self, data, time, event, label, **kwargs):
        
        self.time = time
        self.event = event

        data = data.copy()
        if type(label) == str:
            label = [label]
            data['__label__'] = data[label].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        else:
            data['__label__'] = data[label].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

        kmfs = {}
        
        self.labels = sorted(list(set(data.__label__)))
        self.counts = Counter(data.__label__)
        self.label_names = {}
        self.label_names_list = {}
        self.label_names_size = {}
        for label in self.labels:
            kmf = KaplanMeierFitter()
            ix = data.__label__ == label
            kmfs[label] = kmf.fit(data[ix][time], data[ix][event], label='{}'.format( label ))
            
            # lo = list((kmfs[label].confidence_interval_ -  0.5).abs().sort_values('{}_lower_0.95'.format(label)).index)[0]
            # hi = list((kmfs[label].confidence_interval_ -  0.5).abs().sort_values('{}_upper_0.95'.format(label)).index)[0]

            cis = median_survival_times(kmfs[label].confidence_interval_)
            lo, hi = np.array(cis)[0]
            
            # self.label_names[label] = '{}: N={}; Q2={:.1f}'.format(label, self.counts[label], kmfs[label].median_survival_time_)
            # self.label_names[label] = '{}: N={}; Q2={:.2f} (CI 95% {:.2f} - {:.2f})'.format(label, self.counts[label], kmfs[label].median_survival_time_, lo, hi)
            # self.label_names_list[label] = '{}\tN={}\tQ2={:.2f} (CI 95% {:.2f} - {:.2f})'.format(label, self.counts[label], kmfs[label].median_survival_time_, lo, hi)

            self.label_names[label] = '{}: {} {:.2f} ({:.2f} - {:.2f})'.format(label, self.counts[label], kmfs[label].median_survival_time_, lo, hi)
            self.label_names_list[label] = '{}:\t{}\t{:.2f}\t({:.2f} - {:.2f})'.format(label, self.counts[label], kmfs[label].median_survival_time_, lo, hi)
            self.label_names_size[label] = [len(k) for k in self.label_names_list[label].split('\t')]
        
        self.label_names['__label__'] = ['N Median (95%CI)']
        self.label_names_list['__label__'] = ' \tN\tMedian\t(95% CI)'
        self.label_names_size['__label__'] = [len(k) for k in [' ', 'N', 'Median','(95%CI)']]

        self.data = data[[time, event, '__label__']]
        self.kmfs = kmfs