import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


csv_folder = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/CSV'
feature_list = ['Temp Comp Media', 'Precipitacao']

brazil_stations = ['ARACAJU', 'RECIFE (CURADO)', 'SAO CARLOS', 'ARACUAI', 'ARAGARCAS', 'ARAGUAINA', 'ARAXA', 'ARCOVERDE', 'AREIA', 'ARINOS', 'AVARE', 'AVELAR (P.DO ALFERES)', 'BACABAL', 'BAGE', 'BALSAS', 'BAMBUI', 'BARBACENA', 'BARBALHA', 'BARCELOS', 'BARRA', 'BARRA DO CORDA', 'BARREIRAS', 'BELEM', 'BELO HORIZONTE', 'BELTERRA', 'BENJAMIN CONSTANT', 'BENTO GONCALVES', 'BOA VISTA', 'BOM DESPACHO', 'BOM JESUS', 'BOM JESUS DA LAPA', 'BOM JESUS DO PIAUI', 'BRASILIA', 'BREVES', 'C. DO MATO DENTRO', 'CABROBO', 'CACERES', 'CAETITE', 'CALDAS(P. DE CALDAS)', 'CALDEIRAO', 'CAMETA', 'CAMPINA GRANDE', 'CAMPO MOURAO', 'CAMPOS', 'CAMPOS DO JORDAO', 'CAMPOS NOVOS', 'CAMPOS SALES', 'CANARANA', 'CANAVIEIRAS', 'CAPARAO', 'CAPINOPOLIS', 'CARACARAI', 'CARACOL', 'CARATINGA', 'CARAVELAS', 'CARBONITA', 'CARINHANHA', 'CAROLINA', 'CASTRO', 'CATALAO', 'CATANDUVA', 'CAXIAS', 'CAXIAS DO SUL', 'CEARA MIRIM', 'CHAPADINHA', 'CHAPECO', 'CIPO', 'COARI', 'CODAJAS', 'COLINAS', 'CONCEICAO DO ARAGUAIA', 'CORDEIRO', 'CORONEL PACHECO', 'CORRENTINA', 'CORUMBA', 'CRATEUS', 'CRUZ ALTA', 'CRUZ DAS ALMAS', 'CRUZEIRO DO SUL', 'CRUZETA', 'CUIABA', 'CURITIBA', 'CURVELO', 'DIAMANTINA', 'DIAMANTINO', 'DIVINOPOLIS', 'EIRUNEPE', 'ENCRUZILHADA DO SUL', 'ESPERANTINA', 'ESPINOSA', 'FEIRA DE SANTANA', 'FLORANIA', 'FLORESTAL', 'FLORIANO', 'FLORIANOPOLIS', 'FONTE BOA', 'FORMOSA', 'FORMOSO', 'FORTALEZA', 'FRANCA', 'FRUTAL', 'GARANHUNS', 'GLEBA CELESTE', 'GOIANIA', 'GOIAS', 'GUARAMIRANGA', 'GUARATINGA', 'GUARULHOS', 'IAUARETE', 'IBIRITE', 'IGUATU', 'IMPERATRIZ', 'INDAIAL', 'IPAMERI', 'IRAI', 'IRATI', 'IRECE', 'ITABAIANINHA', 'ITABERABA', 'ITACOATIARA', 'ITAITUBA', 'ITAMARANDIBA', 'ITAPERUNA', 'ITIRUCU (JAGUAQUARA)', 'ITUACU', 'ITUIUTABA', 'ITUMBIARA', 'IVAI', 'IVINHEMA', 'JACOBINA', 'JAGUARUANA', 'JANAUBA', 'JANUARIA', 'JOAO PESSOA', 'JOAO PINHEIRO', 'JUIZ DE FORA', 'JURAMENTO', 'LABREA', 'LAGES', 'LAGOA VERMELHA', 'LAMBARI', 'LAVRAS', 'LENCOIS', 'LONDRINA', 'LUZILANDIA(LAG.DO PIAUI)', 'MACAPA', 'MACAU', 'MACEIO', 'MACHADO', 'MANAUS', 'MANICORE', 'MARABA', 'MARINGA', 'MATUPA', 'MOCAMBINHO', 'MONTE ALEGRE', 'MONTE AZUL', 'MONTE SANTO', 'MONTEIRO', 'MONTES CLAROS', 'MORADA NOVA', 'MORRO DO CHAPEU', 'NATAL', 'NHUMIRIM (NHECOLANDIA)', 'NOVA XAV.(XAVANTINA)', 'OBIDOS', 'OURICURI', 'PADRE RICARDO REMETTER', 'PALMAS', 'PALMEIRA DOS INDIOS', 'PAO DE ACUCAR', 'PARACATU', 'PARANAGUA', 'PARANAIBA', 'PARINTINS', 'PARNAIBA', 'PASSO FUNDO', 'PATOS', 'PATOS DE MINAS', 'PAULISTANA', 'PAULO AFONSO', 'PEDRA AZUL', 'PEDRO AFONSO', 'PEIXE', 'PELOTAS', 'PETROLINA', 'PICOS', 'PIRAPORA', 'PIRENOPOLIS', 'PIRIPIRI', 'POMPEU', 'PONTA PORA', 'PORTO ALEGRE', 'PORTO DE MOZ', 'PORTO DE PEDRAS', 'PORTO NACIONAL', 'POSSE', 'POXOREO', 'PRESIDENTE PRUDENTE', 'PROPRIA', 'QUIXERAMOBIM', 'REMANSO', 'RESENDE', 'RIO BRANCO', 'RIO DE JANEIRO', 'RIO GRANDE', 'RIO VERDE', 'RONCADOR', 'RONDONOPOLIS', 'S.G.DA CACHOEIRA(UAUPES)', 'S?O GONCALO', 'SALINAS', 'SALVADOR (ONDINA)', 'SANTA MARIA', 'SANTA VITORIA DO PALMAR', 'SANTANA DO LIVRAMENTO', 'SAO FELIX DO XINGU', 'SAO JOAO DO PIAUI', 'SAO JOAQUIM', 'SAO JOSE DO RIO CLARO', 'SAO LOURENCO', 'SAO LUIS', 'SAO LUIZ GONZAGA', 'SAO MATEUS', 'SAO PAULO(MIR.de SANTANA)', 'SAO S.DO PARAISO', 'SAO SIMAO', 'SENHOR DO BONFIM', 'SERIDO (CAICO)', 'SERRINHA', 'SETE LAGOAS', 'SOBRAL', 'SOROCABA', 'SOURE', 'STa. R. DE CASSIA (IBIPETUBA)', 'SURUBIM', 'TAGUATINGA', 'TARAUACA', 'TAUA', 'TAUBATE', 'TEFE', 'TERESINA', 'TRACUATEUA', 'TRIUNFO', 'TUCURUI', 'TURIACU', 'UBERABA', 'UNAI', 'URUGUAIANA', 'URUSSANGA', 'VALE DO GURGUEIA (CRISTIANO CASTRO)', 'VICOSA', 'VITORIA', 'VITORIA DA CONQUISTA', 'VOTUPORANGA', 'ZE DOCA']

brazil_stations =['ARACAJU', 'RIO DE JANEIRO', 'SALVADOR (ONDINA)', 'SAO LUIS', 'BELO HORIZONTE', 'MANAUS', 'RECIFE (CURADO)']
missing_percent = []
for station in brazil_stations:


    df =  pd.read_csv(csv_folder+'/{0}.csv'.format(station), index_col=[0])

    missing_dates=df[df['Precipitacao'].isnull() | df['Temp Comp Media'].isnull()]
    print(station, missing_dates)
    missing_percent.append(len(missing_dates))


n_cities = 7
index = range(len(brazil_stations))
ordered_stations = list(zip(*[(s,v) for s,v in sorted(zip(brazil_stations, missing_percent), key = lambda pair: pair[1])]))
fig = plt.figure()
plt.barh(index[0:n_cities], ordered_stations[1][0:n_cities])
plt.yticks(index[0:n_cities], ordered_stations[0][0:n_cities])
fig.set_size_inches([15, n_cities*.8])
plt.savefig(csv_folder+'/missing_data_visualization.png', dpi=100)
