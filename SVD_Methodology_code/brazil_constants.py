EPIDEMIC_CLASSIFICATION_PER_STATION = {
    'Aracaju': [1,1,1,-1,-1,-1,1,1,1,-1,1,1,-1,1,1,-1,-1], #Aracaju: c(2000:2002,2006:2008,2010,2011,2013,2014)
    'BeloHorizonte': [1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,1,1,1,-1], #Belo Horizonte: c(2000,2001,2006:2009, 2012,2013,2014,2015)
     'Manaus': [1,1,1,-1,-1,-1,1,1,-1,1,1,1,1,1,-1,-1,-1], #Manaus: c(2000,2001,2002,2006,2007,2009:2013)
     'Sertaozinho' : [1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,1,1,1,1,-1], #Sertaozinho: c(2000,2001,2005,2006,2009,2010,2012:2015)
     'Barueri':  [-1,1,1,-1,-1,-1,1,-1,-1,1,1,-1,-1,1,1,-1,-1], #Barueri:  c(2001,2002,2006,2009,2010,2013,2014)
     'Barretos': [1,-1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,1,1,-1], #Barretos:  c(2000,2005:2007,2009:2012,2014,2015)
     'BarraMansa': [1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,-1], #Barra Mansa: c(2000:2002,2007,2010,2012,2014,2015)
     'SaoGoncalo':[1,1,-1,-1,-1,-1,1,1,-1,1,1,1,1,-1,1,1,1], #Sao Goncalo: c(2000,2001,2006,2007,2009:2012,2014:2016)
     'SantaCruz':[1,1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,1,1,1], #Santa Cruz: c(2000:2002,2005,2006,2009,2012,2014,2015,2016)
     'Rondonopolis':[-1,1,1,-1,-1,1,-1,-1,1,1,-1,1,1,-1,1,1,-1], #Rondonopolis: c(2001,2002,2005,2008,2009,2011,2012,2014,2015)
     'Parnaiba':[-1,1,1,-1,1,1,1,-1,-1,-1,1,1,1,-1,1,-1,1], #Parnaiba: c(2001,2002,2004:2006,2010:2012,2014,2016)
     'TresLagoas':[1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,1,1,-1], #Tres Lagoas: c(2000,2005,2006,2009:2012,0,2014,2015,0)
     'SaoVicente':[1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,1,1,1,-1,-1], #SaoVicente(2000,2001,2003:2005,2009,2012:2014)
     'Maranguape':[1,1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1], #Maranguape: c(2000:2002,2006,2010:2012,2014)
     'JiParana': [1,-1,-1,-1,1,1,-1,1,1,1,-1,-1,1,1,-1,1,1], #Ji Parana: c(2000,2004,2005,2007:2009,2012,2013,2015,2016)
     'JuazeirodoNorte':[1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,1,-1,-1], #Juazeiro do Norte: c(2000:2002,2004,2006,2009,2011,2014)
     'Guaruja':[1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,1,1,1,1,-1], #Guaruja: c(2000,2001,2004,2005,2009,2012:2015)
     'Eunapolis':[1,1,1,-1,-1,-1,-1,1,1,-1,1,1,1,-1,-1,1,1], #Eunapolis: c(2000:2002,2007,2008,2010:2012,2015,2016)
     'Barra Mansa': [1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,-1], #Barra Mansa: c(2000:2002,2007,2010,2012,2014,2015)
     'SaoLuis':[-1,-1,-1,-1,1,1,1,1,-1,1,1,1,1,-1,1,1,-1], #Sao Luis: c(2004:2007, 2009:2012, 2014:2015)
     'Rio':[1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,1,1,-1] #c(2000,2001,2005,2006,2007,2010,2011,2012,2014,2015)
}


accuracy_window= {
    'Aracaju': .10, #Aracaju: c(2000:2002,2006:2008,2010,2011,2013,2014)
    'BeloHorizonte': .10, #Belo Horizonte: c(2000,2001,2006:2009, 2012,2013,2014,2015)
     'Manaus': .10, #Manaus: c(2000,2001,2002,2006,2007,2009:2013)
     'Sertaozinho' : .10, #Sertaozinho: c(2000,2001,2005,2006,2009,2010,2012:2015)
     'Barueri':  .10, #Barueri:  c(2001,2002,2006,2009,2010,2013,2014)
     'Barretos': .10, #Barretos:  c(2000,2005:2007,2009:2012,2014,2015)
     'BarraMansa': .10, #Barra Mansa: c(2000:2002,2007,2010,2012,2014,2015)
     'SaoGoncalo': .10, #Sao Goncalo: c(2000,2001,2006,2007,2009:2012,2014:2016)
     'SantaCruz': .10, #Santa Cruz: c(2000:2002,2005,2006,2009,2012,2014,2015,2016)
     'Rondonopolis':.10, #Rondonopolis: c(2001,2002,2005,2008,2009,2011,2012,2014,2015)
     'Parnaiba':.17, #Parnaiba: c(2001,2002,2004:2006,2010:2012,2014,2016)
     'TresLagoas':.10, #Tres Lagoas: c(2000,2005,2006,2009:2012,0,2014,2015,0)
     'SaoVicente':.10, #SaoVicente(2000,2001,2003:2005,2009,2012:2014)
     'Maranguape':.10, #Maranguape: c(2000:2002,2006,2010:2012,2014)
     'JiParana': .10, #Ji Parana: c(2000,2004,2005,2007:2009,2012,2013,2015,2016)
     'JuazeirodoNorte': .10, #Juazeiro do Norte: c(2000:2002,2004,2006,2009,2011,2014)
     'Guaruja':.17, #Guaruja: c(2000,2001,2004,2005,2009,2012:2015)
     'Eunapolis':.10, #Eunapolis: c(2000:2002,2007,2008,2010:2012,2015,2016)
     'Barra Mansa': .10, #Barra Mansa: c(2000:2002,2007,2010,2012,2014,2015)
     'SaoLuis':.10, #Sao Luis: c(2004:2007, 2009:2012, 2014:2015)
     'Rio': .17 #c(2000,2001,2005,2006,2007,2010,2011,2012,2014,2015)
}
