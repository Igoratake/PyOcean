from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def TideTable(fname,cte,ano_ini,deltadias,dias,mare,nm):

	# #MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
	# # Previsao de mare astronomica, usando a formulacao
	# #              eta(t)=f*H*np.cos(V+u-G+wt)
	# #
	# #
	# # Referencia:
	# #
	# #    SCHUREMANN, P. 1958.
	# #    Manual of harmonic analysis and prediction of tides.
	# #    Washington, D.C.,U.S. Coast & Geodetic Surv., S.P. n. 98, 317p.
	# #
	# # Componentes utilizadas: Q1,O1,P1,K1,N2,M2,S2,K2,M3,MN4,M4,MS4
	# #MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM

	#Correções de Schuremann
	bk = int(0.25*(ano_ini-1901))
	sk = 277.0247 + (129.38481*(ano_ini-1900)) + (13.17639*(deltadias+bk))
	hk = 280.1895 -   (0.23872*(ano_ini-1900)) +  (0.98565*(deltadias+bk))
	nk = 259.1568 -  (19.32818*(ano_ini-1900)) -  (0.05295*(deltadias+bk))
	pk = 334.3853 +  (40.66249*(ano_ini-1900)) +  (0.11140*(deltadias+bk))
	sk=sk-int(sk/360)*360
	hk=hk-int(hk/360)*360
	nk=nk-int(nk/360)*360
	pk=pk-int(pk/360)*360
	nrad= np.deg2rad(nk)

	#Criando fatores de correção para as constantes
	cte['v0'] = np.nan
	cte['uk'] = np.nan
	cte['fk'] = np.nan
	cte['wk'] = np.nan
	############################################################################################
	#Q1 - 1
	cte['v0']['Q1'] = -(3.*sk)+pk+hk+270
	cte['uk']['Q1'] = (10.8*np.sin(nrad))-(1.34*np.sin(2*nrad)) +  (0.19*np.sin(3*nrad))
	cte['fk']['Q1'] =  1.0089 + (0.1871*np.cos(nrad)) - (0.0147*np.cos(2*nrad))+(0.0014*np.cos(3*nrad))
	cte['wk']['Q1'] = np.deg2rad(13.3986611)
	#O1 - 2
	cte['v0']['O1'] = -(2.*sk)+hk+270
	cte['uk']['O1'] = cte['uk']['Q1']
	cte['fk']['O1'] = cte['fk']['Q1']
	cte['wk'] ['O1']= np.deg2rad(13.9430358)
	#P1 - 3
	cte['v0']['P1'] = -hk + 270
	cte['uk']['P1'] = 0
	cte['fk']['P1'] = 1
	cte['wk']['P1'] = np.deg2rad(14.9589316)
	#K1 - 4
	cte['v0']['K1'] = hk + 90
	cte['uk']['K1'] = -8.86*np.sin(nrad)+(0.08*np.sin(2.*nrad))-(0.07*np.sin(3.*nrad))
	cte['fk']['K1'] =  1.006 + (0.115*np.cos(nrad)) - (0.0088*np.cos(2.*nrad)) + (0.0006*np.cos(3.*nrad))
	cte['wk']['K1'] = np.deg2rad(15.0410689)
	#N2 - 5
	cte['v0']['N2'] = -(3.*sk) + pk + (2.*hk)
	cte['uk']['N2'] = -2.14*np.sin(nrad)
	cte['fk']['N2'] = 1.0004 - (0.0373*np.cos(nrad)) + (0.0002*np.cos(2.*nrad))
	cte['wk']['N2'] = np.deg2rad(28.4397300)
	#M2 - 6
	cte['v0']['M2'] = -(2.*sk) + (2.*hk)
	cte['uk']['M2'] = cte['uk']['N2']
	cte['fk']['M2'] = cte['fk']['N2']
	cte['wk']['M2'] = np.deg2rad(28.9841046)
	#S2 - 7
	cte['v0']['S2'] = 0
	cte['uk']['S2'] = 0
	cte['fk']['S2'] = 1
	cte['wk']['S2'] = np.deg2rad(30.)
	#K2 - 8
	cte['v0']['K2'] = 2. * hk
	cte['uk']['K2'] = -(17.74*np.sin(nrad))+(0.68*np.sin(2.*nrad))-(0.04*np.sin(3.*nrad))
	cte['fk']['K2'] = 1.0241 + (0.2863*np.cos(nrad)) + (0.0083*np.cos(2.*nrad)) - (0.0015*np.cos(3.*nrad))
	cte['wk']['K2'] = np.deg2rad(30.0821378)
	#M3 - 9
	cte['v0']['M3'] = -3.*(sk-hk) + 180
	cte['uk']['M3'] = cte['uk']['M2']*1.5
	cte['fk']['M3'] = cte['fk']['M2']**1.5
	cte['wk']['M3'] = np.deg2rad(43.4761570)
	#MN4 - 10
	cte['v0']['MN4'] = cte['v0']['N2']+cte['v0']['M2']
	cte['uk']['MN4'] = cte['uk']['N2']+cte['uk']['M2']
	cte['fk']['MN4'] = cte['fk']['N2']*cte['fk']['M2']
	cte['wk']['MN4'] = np.deg2rad(57.4238346)
	#M4 - 11
	cte['v0']['M4'] = 2.*cte['v0']['M2']
	cte['uk']['M4'] = cte['uk']['M2']*2
	cte['fk']['M4'] = cte['fk']['M2']**2
	cte['wk']['M4'] = np.deg2rad(57.9682093)
	#MS4 - 12
	cte['v0']['MS4'] = cte['v0']['M2']+cte['v0']['S2']
	cte['uk']['MS4'] = cte['uk']['M2']+cte['uk']['S2']
	cte['fk']['MS4'] = cte['fk']['M2']*cte['fk']['S2']
	cte['wk']['MS4'] = np.deg2rad(58.9841051)
	##########################################################################################################

	horas = dias*24
	elev = []
	for i in list(range(1,horas+1)):
		eta=0;
		for comp in cte.index:
			eta=eta + cte['fk'][comp]*cte['Amplitude'][comp]*np.cos(cte['wk'][comp]*(i-1) +
			np.deg2rad((cte['v0'][comp]+cte['uk'][comp]-cte['Fase'][comp])))

		elev.append(eta)

	mare['Amplitude']= elev
	mare['Amplitude'] = mare['Amplitude']+nm

	if fname != None:

		mare.Amplitude.plot(linestyle=':',linewidth=0.6,color='k')
		plt.ylabel('Elevação (cm)')
		plt.savefig('{}_Mare_Prev.png'.format(fname),dpi=400,bbox='tight')
		plt.close()

		mare.to_excel('{}_Mare_Prev.xlsx'.format(fname))

	return cte, mare

##########################################################################################################
#EXEMPLO DE COMO UTILIZAR

# if __name__ == '__main__':
#
# 	#PERÍODO DE INTERESSE
#
# 	#Data inicial
# 	d0 = date(2019, 12, 08)
#
# 	#Data final
# 	d1 = date(2015, 12, 12)
# 	delta = d1 - d0
# 	dias = delta.days
#
# 	if (d0.month & d0.day) !=1:
# 	    dt = date(d0.year,1,1)
# 	    deltaini = d0 - dt
# 	    deltadias = deltaini.days
# 	else:
# 	    deltadias = 0
#
# 	#Gerando dataframe com Diferença de tempo desejada
# 	mare = pd.DataFrame(index= pd.date_range(d0, d1, freq='1H', closed='left'), columns=['Amplitude'])
#
# 	#CONDIÇÕES INICIAIS
# 	#nivel medio
# 	nm = 77
#
# 	#CONSTANTES
# # COLOCAR AS CONSTANTES SEMPRE EM CM
# 	cte = pd.DataFrame(np.nan, index = ['Q1','O1','P1','K1','N2','M2',
#                                 'S2','K2','M3','MN4','M4','MS4'],
#                                 columns=['Amplitude','Fase'])
# 	#Amp   Fase
# 	#Q1
# 	cte.loc['Q1']= 3.2,33
# 	# #O1
# 	cte.loc['O1']= 9.6,84
# 	# #P1
# 	cte.loc['P1']= 2.2,104
# 	# #K1
# 	cte.loc['K1']= 6.7,106
# 	# #N2
# 	cte.loc['N2']= 4.6,149
# 	# #M2
# 	cte.loc['M2']= 35.3,81
# 	# #S2
# 	cte.loc['S2']= 22.9,75
# 	# #K2
# 	cte.loc['K2']= 6.2,75
# 	# #M3
# 	cte.loc['M3']= 4.5,198
# 	# #MN4
# 	cte.loc['MN4']= 0,0
# 	# #M4
# 	cte.loc['M4']= 1.7,351
# 	# #MS4
# 	cte.loc['MS4']= 1.6,105
#
# 	cte, mare = TideTable(fname = 'teste',cte=cte, ano_ini= d0.year,deltadias = deltadias, dias=dias,mare=mare,nm=nm)
