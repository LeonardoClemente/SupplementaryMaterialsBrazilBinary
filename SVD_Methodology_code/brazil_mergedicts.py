import pickle
import copy

#brazil_stations = ['Aracaju', 'BarraMansa', 'Barretos', 'Barueri', 'BeloHorizonte', 'Eunapolis', 'Guaruja', 'JiParana', 'JuazeirodoNorte', 'Parnaiba', 'Rio', 'Rondonopolis', 'SantaCruz', 'SaoGoncalo', 'Sertaozinho']
#brazil_stations = ['SaoGoncalo', 'SantaCruz', 'JuazeirodoNorte', 'JiParana', 'Rondonopolis', 'Manaus','SaoLuis','BarraMansa', 'Sertaozinho', 'Eunapolis', 'BeloHorizonte', 'Parnaiba', 'SaoVicente', 'Barretos', 'Aracaju', 'Guaruja', 'TresLagoas', 'Maranguape', 'Barueri', 'Rio']
brazil_stations = ['Manaus', 'Aracaju', 'Barueri', 'Sertaozinho', 'BeloHorizonte']
dict_folder = '/Users/leonardo/Desktop/FORECASTING/STOLERMAN/BRAZIL_humidity/heatmap_dicts'

years = list(range(2006,2018))

heatmap_version = 6
for station in brazil_stations:
    heatmap_dict = {}
    for y in years:
        data = pickle.load(open("{0}/{1}/v{2}_{3}.p".format(dict_folder, station, heatmap_version, y), "rb" ))
        heatmap = data[0]
        t0_vector = data[1]
        p_vector = data[2]
        heatmap_dict[y] = copy.copy(heatmap)

    pickle.dump([heatmap_dict, t0_vector, p_vector],open("{0}/{1}/v{2}_ALL_YEARS.p".format(dict_folder, station, heatmap_version, y), "wb" ))
