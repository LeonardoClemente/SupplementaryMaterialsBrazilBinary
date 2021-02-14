import pandas as pd
import numpy as np
import time


brazil_stations = ['ACARAU', 'AGUA BRANCA', 'AIMORES', 'ALAGOINHAS', 'ALTAMIRA', 'ALTO PARNAIBA', 'APODI', 'ARACAJU', 'ARACUAI', 'ARAGARCAS', 'ARAGUAINA', 'ARAXA', 'ARCOVERDE', 'AREIA', 'ARINOS', 'AVARE', 'AVELAR (P.DO ALFERES)', 'BACABAL', 'BAGE', 'BALSAS', 'BAMBUI', 'BARBACENA', 'BARBALHA', 'BARCELOS', 'BARRA', 'BARRA DO CORDA', 'BARREIRAS', 'BELEM', 'BELO HORIZONTE', 'BELTERRA', 'BENJAMIN CONSTANT', 'BENTO GONCALVES', 'BOA VISTA', 'BOM DESPACHO', 'BOM JESUS', 'BOM JESUS DA LAPA', 'BOM JESUS DO PIAUI', 'BRASILIA', 'BREVES', 'C. DO MATO DENTRO', 'CABROBO', 'CACERES', 'CAETITE', 'CALDAS(P. DE CALDAS)', 'CALDEIRAO', 'CAMETA', 'CAMPINA GRANDE', 'CAMPO MOURAO', 'CAMPOS', 'CAMPOS DO JORDAO', 'CAMPOS NOVOS', 'CAMPOS SALES', 'CANARANA', 'CANAVIEIRAS', 'CAPARAO', 'CAPINOPOLIS', 'CARACARAI', 'CARACOL', 'CARATINGA', 'CARAVELAS', 'CARBONITA', 'CARINHANHA', 'CAROLINA', 'CASTRO', 'CATALAO', 'CATANDUVA', 'CAXIAS', 'CAXIAS DO SUL', 'CEARA MIRIM', 'CHAPADINHA', 'CHAPECO', 'CIPO', 'COARI', 'CODAJAS', 'COLINAS', 'CONCEICAO DO ARAGUAIA', 'CORDEIRO', 'CORONEL PACHECO', 'CORRENTINA', 'CORUMBA', 'CRATEUS', 'CRUZ ALTA', 'CRUZ DAS ALMAS', 'CRUZEIRO DO SUL', 'CRUZETA', 'CUIABA', 'CURITIBA', 'CURVELO', 'DIAMANTINA', 'DIAMANTINO', 'DIVINOPOLIS', 'EIRUNEPE', 'ENCRUZILHADA DO SUL', 'ESPERANTINA', 'ESPINOSA', 'FEIRA DE SANTANA', 'FLORANIA', 'FLORESTAL', 'FLORIANO', 'FLORIANOPOLIS', 'FONTE BOA', 'FORMOSA', 'FORMOSO', 'FORTALEZA', 'FRANCA', 'FRUTAL', 'GARANHUNS', 'GLEBA CELESTE', 'GOIANIA', 'GOIAS', 'GUARAMIRANGA', 'GUARATINGA', 'GUARULHOS', 'IAUARETE', 'IBIRITE', 'IGUATU', 'IMPERATRIZ', 'INDAIAL', 'IPAMERI', 'IRAI', 'IRATI', 'IRECE', 'ITABAIANINHA', 'ITABERABA', 'ITACOATIARA', 'ITAITUBA', 'ITAMARANDIBA', 'ITAPERUNA', 'ITIRUCU (JAGUAQUARA)', 'ITUACU', 'ITUIUTABA', 'ITUMBIARA', 'IVAI', 'IVINHEMA', 'JACOBINA', 'JAGUARUANA', 'JANAUBA', 'JANUARIA', 'JOAO PESSOA', 'JOAO PINHEIRO', 'JUIZ DE FORA', 'JURAMENTO', 'LABREA', 'LAGES', 'LAGOA VERMELHA', 'LAMBARI', 'LAVRAS', 'LENCOIS', 'LONDRINA', 'LUZILANDIA(LAG.DO PIAUI)', 'MACAPA', 'MACAU', 'MACEIO', 'MACHADO', 'MANAUS', 'MANICORE', 'MARABA', 'MARINGA', 'MATUPA', 'MOCAMBINHO', 'MONTE ALEGRE', 'MONTE AZUL', 'MONTE SANTO', 'MONTEIRO', 'MONTES CLAROS', 'MORADA NOVA', 'MORRO DO CHAPEU', 'NATAL', 'NHUMIRIM (NHECOLANDIA)', 'NOVA XAV.(XAVANTINA)', 'OBIDOS', 'OURICURI', 'PADRE RICARDO REMETTER', 'PALMAS', 'PALMEIRA DOS INDIOS', 'PAO DE ACUCAR', 'PARACATU', 'PARANAGUA', 'PARANAIBA', 'PARINTINS', 'PARNAIBA', 'PASSO FUNDO', 'PATOS', 'PATOS DE MINAS', 'PAULISTANA', 'PAULO AFONSO', 'PEDRA AZUL', 'PEDRO AFONSO', 'PEIXE', 'PELOTAS', 'PETROLINA', 'PICOS', 'PIRAPORA', 'PIRENOPOLIS', 'PIRIPIRI', 'POMPEU', 'PONTA PORA', 'PORTO ALEGRE', 'PORTO DE MOZ', 'PORTO DE PEDRAS', 'PORTO NACIONAL', 'POSSE', 'POXOREO', 'PRESIDENTE PRUDENTE', 'PROPRIA', 'QUIXERAMOBIM', 'RECIFE (CURADO)', 'REMANSO', 'RESENDE', 'RIO BRANCO', 'RIO DE JANEIRO', 'RIO GRANDE', 'RIO VERDE', 'RONCADOR', 'RONDONOPOLIS', 'S.G.DA CACHOEIRA(UAUPES)', 'S?O GONCALO', 'SALINAS', 'SALVADOR (ONDINA)', 'SANTA MARIA', 'SANTA VITORIA DO PALMAR', 'SANTANA DO LIVRAMENTO', 'SAO CARLOS', 'SAO FELIX DO XINGU', 'SAO JOAO DO PIAUI', 'SAO JOAQUIM', 'SAO JOSE DO RIO CLARO', 'SAO LOURENCO', 'SAO LUIS', 'SAO LUIZ GONZAGA', 'SAO MATEUS', 'SAO PAULO(MIR.de SANTANA)', 'SAO S.DO PARAISO', 'SAO SIMAO', 'SENHOR DO BONFIM', 'SERIDO (CAICO)', 'SERRINHA', 'SETE LAGOAS', 'SOBRAL', 'SOROCABA', 'SOURE', 'STa. R. DE CASSIA (IBIPETUBA)', 'SURUBIM', 'TAGUATINGA', 'TARAUACA', 'TAUA', 'TAUBATE', 'TEFE', 'TERESINA', 'TRACUATEUA', 'TRIUNFO', 'TUCURUI', 'TURIACU', 'UBERABA', 'UNAI', 'URUGUAIANA', 'URUSSANGA', 'VALE DO GURGUEIA (CRISTIANO CASTRO)', 'VICOSA', 'VITORIA', 'VITORIA DA CONQUISTA', 'VOTUPORANGA', 'ZE DOCA']

#Turning INMET's format to a legible CSV!


file_name = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/TXT/aracaju.txt'
output_folder = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/CSV'

feature_list = ['Temp Comp Media', 'Precipitacao']

for station in brazil_stations:

    df=pd.read_csv('/Users/leonardo/Desktop/FORECASTING/STOLERMAN/TXT/{0}.txt'.format(station), sep=';',skiprows=16)
    df['Data'] =pd.to_datetime(df['Data'], format='%d/%m/%Y')

    all_dates = pd.date_range(start='2001-01-01', end='2015-12-31').strftime("%Y-%m-%d")

    missing_dates_0 = []
    missing_dates_1200 = []
    df.set_index(df['Data'], inplace=True)


    feature_dict = dict(zip(feature_list, [[] for feature in feature_list]))
    #Generate dataframe

    for date in all_dates:

        for feature in feature_list:

            if date not in df[df['Hora'] == 0].index:
                missing_dates_0.append(date)
                f0=float('nan')
            else:
                f0=df[df['Hora'] == 0][feature][date]

            if date not in df[df['Hora'] == 1200].index:
                missing_dates_1200.append(date)
                f1200= float('nan')
            else:
                f1200=df[df['Hora'] == 1200][feature][date]

            isnanf0 = np.isnan(f0)
            isnanf1200 = np.isnan(f1200)

            if isnanf0 and isnanf1200:
                feature_dict[feature].append(f0)
            elif isnanf0 and not isnanf1200:
                feature_dict[feature].append(f1200)
            elif not isnanf0 and isnanf1200:
                feature_dict[feature].append(f0)
            elif not (isnanf0 + isnanf1200):
                print('{3} {2} valid values on both hours encountered: \n 0= {0} \
                      \n 1200:{1}'.format(f0,f1200, date, station))
                feature_dict[feature].append(np.max([f0,f1200]))



    new_df = pd.DataFrame(feature_dict, index=all_dates)
    new_df.to_csv(output_folder+'/{0}.csv'.format(station))
