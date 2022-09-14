# -*- coding: utf-8 -*-

import glob
import os
import xlsxwriter
from numpy import asarray, max, arange, round, insert, radians
from numpy import ceil, ma, cumsum, array, argmin
from numpy import linspace, meshgrid, histogram2d, flipud, size, sum
from numpy import nanmax, nanmean, nansum
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy.interpolate import griddata
from scipy import interpolate, ndimage
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from TTutils.logo import *
# Importando o seaborn para padrozinacao
import seaborn as sns

sns.set_style("whitegrid", {'grid.color': "black",
                            'axes.edgecolor': "black",
                            }
              )
sns.set_context('notebook', font_scale=1.7)

# histograma direcional


def HistDir(P, D, arqname='', Pmax=None, MaxProb=None, par='hs', MyRange=[],
            dir16=True, interpolado=True, r_angle=None, porcentagem=False,
            format_cond=False,table_complete=False,version2=False,logo=True):
    '''
    Makes a directional histogram for a single point

    P = array/DataFrame(obrigatorio)
        intensidade
    D = array/DataFrame (obrigatorio)
        direção em graus
    arqname =  string (opcional default '')
        string com o nome do output desejado
    Pmax =  None/float(opcional default None)
        Caso queira fixar o eixo de velocidades entre com o valor de um
        inteiro, caso não use None e o valor sera calculado automaticamente
    MaxProb = None/int (opcional default None)
        Caso queira fixar o eixo de probabilidades entre com o valor de
        um float, caso não use None e o valor sera calculado automaticamente
    par = string (opcional default 'hs')
        defina o parametro que esta sendo analisado
    MyRange = list (opcional)
        entre com uma lista dos ranges de insidade, caso não entrar ele farão
        o calculo automático segundo o parametro
    dir16 = True/False (opcional)
        Se as direçõeses serão divididas em 16 ou 8 (default 16 (True))
    interpolado = True/False (opcional)
        Se desejar gerar um histograma interpolado ou rosa
        (default histograma interpolado (True))
    r_angle = None/float (opicional)
        Se desejar alterar a posiçãoo dos labels dos angulos entre com o
        float do angulo, caso nao None (default None)
    porcentagem True/False (opcional)
        Se desejar que o histograma calcule a porcentagem ou as ocorrencias
        (apenas na tabela)
    format_cond = True/False (opcional, default False)
        Plota a tabela com valores máximos em vermelho
    table_complete = True/False (opcional, default False)
        Força Máximo na tabela (T_bins)
    version2 = True/False (opcional, default False)
        Formato da tabela atualizado, com outros detalhes.

    Detalhes :
        Parametros - escolha dependendo da analise
                hs = Altura significativa
                tp = Período de onda
                corrente = corrente
                vento = vento
                energia = energia de onda

        *** Os dados não devem conter NaN ***

    output:
        A png figure and excel table with directional histogram analysis
    '''

    # Transforma as entradas para numpy arrays
    P = asarray(P)
    D = asarray(D)
    # verificar possivel erro do usuario
    if len(D.shape) > 1:
        if D.shape[1] == 1:
            D = D[:, 0]
        elif D.shape[0] == 1:
            D = D[0, :]
        else:
            print(u'Verifique as dimensões da direção')
            sys.pause('')
    if len(P.shape) > 1:
        if P.shape[1] == 1:
            P = P[:, 0]
        elif P.shape[0] == 1:
            P = P[0, :]
        else:
            print(u'Verifique as dimensões do parametro de entrada')
    # verifica se o valor de m�ximo esta definido
    if Pmax is None:
        Pmax = max(P)
    # verifica o parametro para estabelecer os dados de entrada
    if par == 'hs':
        # titulo do plot
        titulo2 = u'Altura de onda (m) - convenção meteorológica'
        # parte do cabeçalho da tabela de ocorrencia conjunta
        cabecalho = u'Altura (m)'
        # parte do nome do arquivo
        fname = arqname + '_altura'
        myfmt = '0.0'
        # define o numero de divisões do parametro na tabela de ocorrencia
        # verifica se há de 5 a 10 classes
        P_bins = arange(0, round(Pmax) + 0.5, 0.5)
        if len(P_bins) > 11:
            P_bins = arange(0, round(Pmax) + 1, 1)
        if table_complete==True:
            T_bins = arange(0, round(max(P)) + 0.5, 0.5)
            if len(T_bins) > 11:
                T_bins = arange(0, round(max(P)) + 1, 1)            
    elif par == 'tp':
        titulo2 = u'Período de onda (s' + u') - convenção meteorológica'
        cabecalho = u'Período (s)'
        fname = arqname + '_periodo'
        myfmt = '0.0'
        P_bins = arange(0, round(Pmax) + 2, 2)
        if len(P_bins) > 11:
            P_bins = arange(0, round(Pmax) + 3, 3)
        elif len(P_bins) < 6:
            P_bins = arange(0, round(Pmax) + 1, 1)
        if table_complete==True:
            T_bins = arange(0, round(max(P)) + 2, 2)
            if len(T_bins) > 11:
                T_bins = arange(0, round(max(P)) + 3, 3)
            elif len(T_bins) < 6:
                T_bins = arange(0, round(max(P)) + 1, 1)
    elif par == 'vento':
        titulo2 = u'dos Ventos - convenção meteorológica'
        cabecalho = u'Vel. (m/s)'
        fname = arqname + '_vento'
        myfmt = '0.0'
        P_bins = arange(0, Pmax + 2, 2)
        if len(P_bins) > 11:
            if len(P_bins) > 11:
                P_bins = arange(0, Pmax + 2.5, 2.5)
                if len(P_bins) > 11:
                    P_bins = arange(0, Pmax + 5, 5)
        elif len(P_bins) < 6:
            P_bins = arange(0, Pmax + 1.0, 1.0)
            if len(P_bins) < 6:
                P_bins = arange(0, Pmax + 1.0, 1.0)
        if table_complete==True:
            T_bins = arange(0, max(P) + 2, 2)
            if len(T_bins) > 11:
                if len(T_bins) > 11:
                    T_bins = arange(0,  max(P) + 2.5, 2.5)
                    if len(T_bins) > 11:
                        T_bins = arange(0,  max(P) + 5, 5)
            elif len(T_bins) < 6:
                T_bins = arange(0, 10. + 1, 1)
                if len(T_bins) < 6:
                    T_bins = arange(0,  max(P) + 0.5, 0.5)
    elif par == 'energia':
        titulo2 = u'Energia de onda (kJ/m²) - convenção meteorológica'
        cabecalho = u'Energia (kJ/m²)'
        fname = arqname + '_energia'
        myfmt = '0.0'
        P_bins = arange(0, round(Pmax / 10.) * 10. + 2, 2)
        if len(P_bins) > 11:
            P_bins = arange(0, round(Pmax / 10.) * 10. + 3, 3)
        if len(P_bins) > 11:
            P_bins = arange(0, round(Pmax / 10.) * 10. + 5, 5)
        elif len(P_bins) < 6:
            P_bins = arange(0, round(Pmax / 10.) * 10. + 1, 1)
        if table_complete==True:            
            T_bins = arange(0, round(max(P) / 10.) * 10. + 2, 2)
            if len(T_bins) > 11:
                T_bins = arange(0, round(max(P) / 10.) * 10. + 3, 3)
            if len(T_bins) > 11:
                T_bins = arange(0, round(max(P) / 10.) * 10. + 5, 5)
            elif len(T_bins) < 6:
                T_bins = arange(0, round(max(P) / 10.) * 10. + 1, 1)
    elif par == 'corrente':
        titulo2 = u'Intensidade da Corrente (m/s) - convenção  oceanográfica'
        cabecalho = u'Corrente(m/s)'
        fname = arqname + '_corrente'
        myfmt = '0.00'
        P_bins = arange(0, round(Pmax * 10.) / 10. + 0.1, 0.1)
        if len(P_bins) > 11:
            if len(P_bins) > 11:
                P_bins = arange(0, round(Pmax * 10.) / 10. + 0.2, 0.2)
                if len(P_bins) > 11:
                    P_bins = arange(0, round(Pmax * 10.) / 10. + 0.25, 0.25)
        elif len(P_bins) < 6:
            P_bins = arange(0, round(Pmax * 10) / 10. + 0.05, 0.05)
            if len(P_bins) < 6:
                P_bins = arange(0, round(Pmax * 10) / 10. + 0.03, 0.03)
        if table_complete==True:
            T_bins = arange(0, round(max(P) * 10.) / 10. + 0.1, 0.1)
            if len(T_bins) > 11:
                if len(T_bins) > 11:
                    T_bins = arange(0, round(max(P) * 10.) / 10. + 0.2, 0.2)
                    if len(T_bins) > 11:
                        T_bins = arange(0, round(max(P) * 10.) / 10. + 0.25, 0.25)
            elif len(T_bins) < 6:
                T_bins = arange(0, round(max(P) * 10) / 10. + 0.05, 0.05)
                if len(P_bins) < 6:
                    T_bins = arange(0, round(max(P) * 10) / 10. + 0.03, 0.03)
    else:
        print(u'defina um dos parâmetro \n hs = Altura significativa \n \
        tp = Período de onda \n corrente = corrente \n vento = vento \n \
        energia = energia de onda')
    #Fix P_bins
    P_bins=np.round(P_bins,2)
    # caso o usuario tenha definido o range
    if MyRange != []:
        del P_bins
        P_bins = arange(0, Pmax + MyRange, MyRange)
    # verifica se serão 16 ou 8 direções (se aplica pra tabela)
    if dir16 is True:
        # subtrair da direção
        dirdiff = 11.25
        # numero de bins da direção
        dirrg = 17
        # cabeçalho de direções da tabela
        head = [cabecalho, 'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', u'(%)']
    else:
        dirdiff = 22.5
        dirrg = 9
        head = [cabecalho, 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', u'(%)']
    # plota o polar
    ax = plt.subplot(1, 1, 1, polar=True)
    # define que o zero sera no norte
    ax.set_theta_zero_location("N")
    # roda o grafico azimuital
    ax.set_theta_direction(-1)
    ax.yaxis.grid(color='#898989')
    ax.xaxis.grid(color='#898989')
    if interpolado:
        dir_bins = linspace(-dirdiff, 360 - dirdiff, dirrg)
        if np.any(D < 0):
            D[D < 0] += 360
        D[D > (360 - dirdiff)] -= 360
        P_bins_hist = linspace(0, Pmax, 10)
        # calcular o histograma 2d baseado nos bins
        table = (histogram2d(x=P,
                             y=D + dirdiff,
                             bins=[P_bins_hist, dir_bins],
                             normed=True))
        binarea = P_bins_hist[1] * dir_bins[1] * 200
        # grade para plot
        x, y = meshgrid(table[2][:], table[1][:])
        # calcular a porcentagem
        z = table[0] * binarea
        lowlev = z.max() * 0.05
        z[z < lowlev] = -lowlev * 0.9
        z = insert(z, z.shape[1], (z[:, 0]), axis=1)
        # Duplica ultima coluna para completar o diag polar
        z = insert(z, z.shape[0], (z[-1]), axis=0)
        # adiciona os zeros para que a interpolação não estoure o gráfico
        # converte para radianos
        x = radians(x)

        # interpola todos os parametros
        z = ndimage.zoom(z, 5)
        x = ndimage.zoom(x, 5)
        y = ndimage.zoom(y, 5)
        # calcular a porcentagem
        # definir a prob max
        if MaxProb is None:
            MaxProb = z.max() * 1.1
        # plota dados,define os limites do grafico, palheta de cores, a origem
        cax = ax.contourf(x,
                          y,
                          z,
                          linspace(0, ceil(MaxProb), 200),
                          cmap=plt.cm.jet,
                          origin='lower',
                          antialiased=False)
        # numeros e a distância do grafico polar
        ff = ceil(MaxProb / 10) + 1
        cb = plt.colorbar(cax,
                          pad=.075,
                          shrink=0.8,
                          format='%i',
                          ticks=arange(0, ceil(MaxProb) + ff, ff))
        cb.ax.set_title(u'(%)\n')
        cb.set_clim(0, ceil(MaxProb))
        # define a cor em zero
        cb.vmin = 0
        # tipo
        tipo = u'_histograma_direcional.png'
        # titulo
        titulo1 = u'Histograma direcional -'
    else:

        try:
            P = ma.masked_equal(P, 0)
            D = ma.masked_array(D, P.mask)
            P = ma.MaskedArray.compressed(P)
            D = ma.MaskedArray.compressed(D)
        except BaseException:
            pass

        dir_bins = linspace(-dirdiff, 360 - dirdiff, dirrg)
        if np.any(D < 0):
            D[D < 0] += 360
        if np.any(D > 360):
            D[D > 360] -= 360
        if np.any(D > (360 - dirdiff)):
            D[D > (360 - dirdiff)] -= 360
        # calcular o histograma 2d baseado nos bins
        table = (histogram2d(x=P,
                             y=D,
                             bins=[P_bins, dir_bins],
                             normed=False)
                 )
        theta = radians(table[2][:])[:-1]

        stat = cumsum(table[0], axis=0) / table[0].sum() * 100.
        legenda = []

        windcolors = flipud(
            plt.cm.hsv(
                list(map(
                    int, list(
                        round(
                            linspace(0, 180, len(P_bins)
                                     )
                        )
                    )
                )
                )
            )
        )
        for k in flipud(range(size(stat, 0))):
            ax.bar(theta + abs(np.min(theta)),
                   stat[k, :],
                   width=radians(dirdiff * 2),
                   bottom=0.0,
                   color=windcolors[k],
                   edgecolor="k")
            legenda.append('-'.join([str(table[1][k]), str(table[1][k + 1])]))
        if MaxProb is not None:
            ax.set_rmax(MaxProb)
        ax.tick_params(direction='out', length=4, color='r', zorder=10,labelsize=12)
        legenda[-1] = u'>' + str(table[1][k])
        lg = plt.legend(legenda, bbox_to_anchor=(1.6, .9), title=cabecalho, prop={'size': 12})
        lg.get_title().set_fontsize(12)
        tipo = u'_rosa_direcional.png'
        if (par == 'vento'):
            titulo1 = 'Rosa '
        else:
            titulo1 = 'Rosa de '
    # ajustar eixo automaticamente
    if r_angle is None:
        r_angle = dir_bins[argmin(sum(table[0], axis=0))]
    ax.set_rlabel_position(r_angle)
    if logo == True:
        # carregar o logo
        a = plt.axes([.1, .02, .2, .2], facecolor='None')
        im = plt.imshow(array(Image.open(GetLogo())))
        plt.axis('off')
        plt.setp(a, xticks=[], yticks=[])
    # titulo do gráfico
    ax.set_title(titulo1 + titulo2, fontsize=11, y=1.1)
    if arqname == '':
        plt.show()
    else:
        # nome da figura
        plt.savefig(fname + tipo, format='png', dpi=300, bbox_inches='tight')
    # limpa a figura
    plt.clf()
    plt.cla()
    plt.close()
    print('Figure Done\n')

    if arqname != '':
        D[D > 360] -= 360
        D[D > (360 - dirdiff)] -= 360
        dir_bins = linspace(0, 360, dirrg) - dirdiff
        if table_complete==False:
            T_bins = P_bins
        # calcular o histograma 2d baseado nos bins
        table = (histogram2d(x=P,
                             y=D,
                             bins=[T_bins, dir_bins],
                             normed=porcentagem)
                 )
        if porcentagem:
            binarea = T_bins[1] * dir_bins[1] * 200
        else:
            binarea = 1
        # escrever xlsx de sa�da
        workbook = xlsxwriter.Workbook(fname + u'_ocorrencia_conjunta.xlsx')
        # da o nome do ponto para a tabela
        worksheet = workbook.add_worksheet()
        # informa��es para formata��o
        # tamanho das colunas em cm
        worksheet.set_column('A:B', 14)
        worksheet.set_column('B:S', 5)

        if version2==False:
            # criar formatos (ja inserindo negrito)
            format1 = workbook.add_format({'bold': True,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#C9C9C9'})
            format2 = workbook.add_format({'bold': False,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#FFFFFF'})
            format3 = workbook.add_format({'bold': False,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#FFFFFF'})
            format4 = workbook.add_format({'bold': False,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#C9C9C9'})
            format5 = workbook.add_format({'bold': False,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#C9C9C9'})
            format6 = workbook.add_format({'bold': True,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#C9C9C9'})
            format7 = workbook.add_format({'bold': False,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#C9C9C9'})
            format8 = workbook.add_format({'bold': True,
                                        'font_name': 'Arial','font_size':10,
                                        'align':'center','font_color':'#FF0000'})
            # formata��o das casas decimais
            if porcentagem:
                format2.set_num_format('0.00')
            else:
                format2.set_num_format('0')
            format3.set_num_format('0.0')
            format5.set_num_format(myfmt)
            format7.set_num_format(myfmt)
            # inserir linhas de divis�o da c�lula
            format4.set_top(1)
            format6.set_bottom(1)
            format7.set_bottom(1)

            # insere o cabeçalho no arquivo
            for k, hd in enumerate(head):
                worksheet.write(0, k, hd, format6)
            # escreve as linhas de ocorrencia
            for j in range((len(table[1]) - 1)):
                worksheet.write(
                    j + 1,
                    0,
                    '-'.join([
                        str(table[1][j]),
                        str(table[1][j + 1])]
                    ).replace('.', ','),
                    format1)

                for i in range(len(head) - 2):
                    worksheet.write(
                        j + 1, i + 1, table[0][j, i] * binarea, format2)
                if porcentagem:
                    worksheet.write(j +
                                    1, i +
                                    2, np.sum(table[0][j, :]) *
                                    binarea, format3)
                else:
                    worksheet.write(
                        j + 1,
                        i + 2,
                        np.sum(table[0][j, :]) / np.sum(table[0]) * 100,
                        format3)
            # escreve o total e a porcentagem de valores por direção
            worksheet.write(j + 2, 0, u'(%)', format1)
            totais = np.sum(table[0], axis=0) * binarea
            if porcentagem is False:
                totais = totais / np.sum(totais) * 100
            for i, total in enumerate(totais):
                worksheet.write(j + 2, i + 1, total, format5)

            worksheet.write(j + 2, i + 3, '', format5)
            worksheet.write(j + 2, i + 2, '', format5)
            worksheet.write(j + 3, i + 3, '', format5)
            worksheet.write(j + 3, i + 2, '', format5)
            worksheet.write(j + 4, i + 3, '', format7)
            worksheet.write(j + 4, i + 2, '', format7)

            # escreve as medias e os maximos de cada direção
            worksheet.write(j + 3, 0, u'Media', format5)
            worksheet.write(j + 4, 0, u'Max.', format7)
            if np.ma.isMaskedArray(P) is False:
                P = np.ma.masked_object(P, -99999)
            for l in range(len(table[2]) - 1):
                if len(np.ma.MaskedArray.compressed(
                        P[(D > table[2][l]) & (D < table[2][l + 1])])) == 0:
                    worksheet.write(j + 4, l + 1, 0, format7)
                    worksheet.write(j + 3, l + 1, 0, format5)
                else:
                    worksheet.write(
                        j + 4, l + 1, np.nanmax(
                            P[(D > table[2][l]) & (D < table[2][l + 1])]
                            ),
                        format7)
                    worksheet.write(
                        j + 3, l + 1, np.nanmean(
                            P[(D > table[2][l]) & (D < table[2][l + 1])]
                            ),
                        format5)
            
            # Plota tabela com os máximos de direção e velocidade em vermelho
            if format_cond==True:
                if dir16==True:
                    worksheet.conditional_format('R2:R6', {'type': 'top',
                                                'value': '1',
                                                'format': format8})

                    worksheet.conditional_format('B8:Q8', {'type': 'top',
                                                'value': '1',
                                                'format': format8})

                    worksheet.conditional_format('B9:Q9', {'type': 'top',
                                                'value': '1',
                                                'format': format8})

                    worksheet.conditional_format('B10:Q10', {'type': 'top',
                                                'value': '1',
                                                'format': format8})
                    
                else:
                    worksheet.conditional_format('J2:J6', {'type': 'top',
                                                'value': '1',
                                                'format': format8})

                    worksheet.conditional_format('B8:I8', {'type': 'top',
                                                'value': '1',
                                                'format': format8})

                    worksheet.conditional_format('B9:I9', {'type': 'top',
                                                'value': '1',
                                                'format': format8})

                    worksheet.conditional_format('B10:I10', {'type': 'top',
                                                'value': '1',
                                                'format': format8})
        
        else:    
            # criar formatos (ja inserindo negrito)
            format1 = workbook.add_format({'bold': True,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#F0F0F0'})
            format1b = workbook.add_format({'bold': True,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#FFFFFF'})
            format2 = workbook.add_format({'bold': False,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#FFFFFF'})
            format3 = workbook.add_format({'bold': False,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#FFFFFF'})
            format4 = workbook.add_format({'bold': False,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#F0F0F0'})
            format5 = workbook.add_format({'bold': False,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#F0F0F0'})
            format5b = workbook.add_format({'bold': False,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#F0F0F0'})                                       
            format6 = workbook.add_format({'bold': True,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#F0F0F0'})
            format7 = workbook.add_format({'bold': False,
                                        'font_name': 'Arial', 'font_size': 10,
                                        'align': 'center',
                                        'bg_color': '#F0F0F0'})
            format8 = workbook.add_format({'bold': True,
                                        'font_name': 'Arial','font_size':10,
                                        'align':'center','font_color':'#FF0000'})
            # formata��o das casas decimais
            if porcentagem:
                format2.set_num_format('0.00')
            else:
                format2.set_num_format('0')
            format3.set_num_format('0.0')
            format5.set_num_format(myfmt)
            format7.set_num_format(myfmt)
            # inserir linhas de divis�o da c�lula
            format1.set_top(1)
            format4.set_top(1)
            format5b.set_top(1)
            format6.set_bottom(1)
            format6.set_top(1)
            format7.set_bottom(1)
            
            # insere o cabeçalho no arquivo
            for k, hd in enumerate(head):
                worksheet.write(0, k, hd, format6)
            # escreve as linhas de ocorrencia
            for j in range((len(table[1]) - 1)):
                worksheet.write(
                    j + 1,
                    0,
                    '-'.join([
                        str(table[1][j]),
                        str(table[1][j + 1])]
                    ).replace('.', ','),
                    format1b)

                for i in range(len(head) - 2):
                    worksheet.write(
                        j + 1, i + 1, table[0][j, i] * binarea, format2)
                if porcentagem:
                    worksheet.write(j +
                                    1, i +
                                    2, np.sum(table[0][j, :]) *
                                    binarea, format3)
                else:
                    worksheet.write(
                        j + 1,
                        i + 2,
                        np.sum(table[0][j, :]) / np.sum(table[0]) * 100,
                        format3)
            # escreve o total e a porcentagem de valores por direção
            worksheet.write(j + 2, 0, u'(%)', format1)
            totais = np.sum(table[0], axis=0) * binarea
            if porcentagem is False:
                totais = totais / np.sum(totais) * 100
            for i, total in enumerate(totais):
                worksheet.write(j + 2, i + 1, total, format5b)

            worksheet.write(j + 2, i + 2, '', format5b)
            worksheet.write(j + 3, i + 2, '', format5)
            worksheet.write(j + 4, i + 2, '', format7)

            # escreve as medias e os maximos de cada direção
            worksheet.write(j + 3, 0, u'Media', format5)
            worksheet.write(j + 4, 0, u'Max.', format7)
            if np.ma.isMaskedArray(P) is False:
                P = np.ma.masked_object(P, -99999)
            for l in range(len(table[2]) - 1):
                if len(np.ma.MaskedArray.compressed(
                        P[(D > table[2][l]) & (D < table[2][l + 1])])) == 0:
                    worksheet.write(j + 4, l + 1, 0, format7)
                    worksheet.write(j + 3, l + 1, 0, format5)
                else:
                    worksheet.write(
                        j + 4, l + 1, np.nanmax(
                            P[(D > table[2][l]) & (D < table[2][l + 1])]
                            ),
                        format7)
                    worksheet.write(
                        j + 3, l + 1, np.nanmean(
                            P[(D > table[2][l]) & (D < table[2][l + 1])]
                            ),
                        format5)

        # encerra o arquivo
        workbook.close()