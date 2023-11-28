import discord
from discord import Webhook
from discord.ext import commands, tasks
import datetime
from discordwebhook import Discord # Configuración del bot
import aiohttp
import asyncio


#Parametros iniciales del robot --------------------------------------------------------------------------------------
token = 'YOUR_TOKEN'
prefix = '!'

# Crear instancia del bot con Intents
intents = discord.Intents.default()
intents.all()

bot = commands.Bot(command_prefix=prefix, intents=intents)

tiempo_ini = datetime.timedelta(hours=0, minutes=39)
canal_notificaciones_id = 'YOUR_CHANNEL_ID'

webhook_url = 'YOUR_WEBHOOK_URL'
# Variable para almacenar el tiempo de inicio del bot
start_time = datetime.datetime.now()
num_runs_reaper = 0

#%% COMANDOS RUN MINATO --------------------------------------------------------------------------------------
# Función para el temporizador
async def temporizador():
    global tiempo_ini, start_time, num_runs_reaper

    while True:
        tiempo_transcurrido = datetime.datetime.now() - start_time
        tiempo_restante = max(tiempo_ini - tiempo_transcurrido, datetime.timedelta(0))

        if tiempo_restante == datetime.timedelta(0):
            tiempo_ini = datetime.timedelta(minutes=59)  # Restablecer tiempo_ini a 2 minutos
            start_time = datetime.datetime.now()  # Restablecer el tiempo de inicio
            num_runs_reaper += 1
            print(f"Temporizador reseteado. Runs totales de D-REAPER: {num_runs_reaper}")

        await asyncio.sleep(10)  # Esperar 10 segundos antes de volver a verificar

# Comando para iniciar el temporizador y obtener el tiempo restante
@bot.command(name='dr', help='Actualiza la cuenta regresiva Minato Run.')
async def dr(ctx):
    global start_time, tiempo_ini, num_runs_reaper

    # Iniciar el temporizador en segundo plano si aún no se ha iniciado
    if not any(task.get_name() == 'temporizador' for task in asyncio.all_tasks()):
        bot.loop.create_task(temporizador())

    # Calcular la diferencia de tiempo desde la última ejecución
    tiempo_transcurrido = datetime.datetime.now() - start_time
    tiempo_restante = max(tiempo_ini - tiempo_transcurrido, datetime.timedelta(0))

    horas, minutos = divmod(tiempo_restante.seconds // 60, 60)

    await ctx.send(f"Tiempo restante para Minato Run: {horas} horas {minutos} minutos. El detalle es: {tiempo_restante}")



# Definir la tarea para verificar y notificar cada minuto
@tasks.loop(minutes=1)
async def verificar_tiempo_reaper():
    global tiempo_ini, start_time

    # Calcular la diferencia de tiempo desde la última ejecución
    tiempo_transcurrido = datetime.datetime.now() - start_time
    tiempo_restante = max(tiempo_ini - tiempo_transcurrido, datetime.timedelta(0))

    if datetime.timedelta(minutes=9, seconds=10) < tiempo_restante <= datetime.timedelta(minutes=10):
        await notificar_reaper_10()
        print("¡Notificación enviada!")

    if datetime.timedelta(minutes=4, seconds=10) < tiempo_restante <= datetime.timedelta(minutes=5):
        await notificar_reaper_5()
        print("¡Notificación enviada!")

    if datetime.timedelta(minutes=0, seconds=30) < tiempo_restante <= datetime.timedelta(minutes=1):
        #tiempo_ini = datetime.timedelta(hours=0, minutes=11)
        await notifica_inicio_dreaper()
        print("¡Notificación enviada!")







# Comando para actualizar el tiempo inicial de minato_run
@bot.command(name='actualizar_tiempo_ini_reaper', help='Actualiza el tiempo inicial de Dreaper_run.')
@commands.is_owner()
async def actualizar_tiempo_ini_reaper(ctx, tiempo_str):
    global tiempo_ini, start_time
    try:
        tiempo_ini = datetime.timedelta(hours=int(tiempo_str.split('H')[0]), minutes=int(tiempo_str.split('M')[0].split('H')[1]))
        start_time = datetime.datetime.now()  # Reiniciar el tiempo de inicio
        await ctx.send(f"Tiempo inicial actualizado a: {tiempo_ini}.")
    except ValueError:
        await ctx.send("Formato de tiempo incorrecto. El formato debe ser HHMM, por ejemplo, 1H26M.")




# Función para verificar si el comando se ejecuta en el canal correcto
def check_channel(ctx):
    return ctx.channel.id == int(canal_notificaciones_id)



# Función para enviar una notificación a través de un webhook
async def notifica_inicio_dreaper():
    async with aiohttp.ClientSession() as session:
        webhook = Webhook.from_url(webhook_url, session=session)
        embed = discord.Embed(title='Queda 1 minuto aprox para D-Reaper Ch0')
        await webhook.send(embed = embed, username= 'Dukemon')

async def notificar_reaper_10():
    async with aiohttp.ClientSession() as session:
        webhook = Webhook.from_url(webhook_url, session=session)
        embed = discord.Embed(title='Quedan ~10 minutos para Run de D-Reaper Ch0')
        await webhook.send(embed = embed, username= 'Dukemon')

# Función para enviar una notificación a través de un webhook
async def notificar_reaper_5():
    async with aiohttp.ClientSession() as session:
        webhook = Webhook.from_url(webhook_url, session=session)
        embed = discord.Embed(title='Quedan ~5 minutos para Run de D-Reaper Ch0')
        await webhook.send(embed = embed, username= 'Dukemon')


# Función para notificar que el bot se está ejecutando
async def notificar_inicio():
    async with aiohttp.ClientSession() as session:
        webhook = Webhook.from_url(webhook_url, session=session)
        embed = discord.Embed(title='Estoy en línea para informarles del tiempo aproximado de las runs')
        await webhook.send(embed = embed, username= 'Dukemon')


# Evento de inicio del bot
@bot.event
async def on_ready():
    print(f'Bot conectado como {bot.user.name}')
    await notificar_inicio()
    verificar_tiempo_reaper.start()
# Iniciar el bot
bot.run(token)



