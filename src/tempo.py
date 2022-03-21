
from datetime import datetime

def inicio():
    start_time = datetime.now()
    print("Início: {}".format(datetime.strftime(start_time, "%d/%m/%Y %H:%M:%S")))
    return start_time

def deltacomponents(tdelta):
    d = {"dias": tdelta.days}
    d["horas"], rem = divmod(tdelta.seconds, 3600)
    d["min"], d["seg"] = divmod(rem, 60)
    d["total_min"] = tdelta.seconds/60
    d["total_seg"] = tdelta.seconds
    return d

def strfdelta(tdelta, fmt):
    """Exemplo: "Tempo: {}".format(strfdelta(elapsed_time, "{dias} dias, {horas} hrs, {min} min, {seg} seg"))
    """
    components = deltacomponents(tdelta)
    return fmt.format(**components)

def fim(start_time, msg=None):
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Fim: {}".format(datetime.strftime(end_time, "%d/%m/%Y %H:%M:%S")))
    return deltacomponents(elapsed_time)