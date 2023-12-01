import os
import discord
import response

async def response_message(message, user_message, is_private):
    try:
        respond = response.get_response(user_message, str(message.author))
        await message.author.send(respond) if is_private else await message.channel.send(respond)
    except Exception as e:
        print(e)


def run_discord_bot():
    TOKEN = os.environ.get("DISCORD_TOKEN")
    main_channel_id = int(os.environ.get("MAIN_CHANNEL_ID"))

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        global main_channel
        main_channel = client.get_channel(main_channel_id)
        print(f'{client.user} is online')

    # Send Message when member joined a channel
    @client.event
    async def on_voice_state_update(member, before, after):
        # Check if the member joined a voice channel
        if before.channel is None and after.channel is not None:
            # send greeting message at main channel
            channel_name = after.channel.name
            await main_channel.send(f'Welcome to {channel_name}, {member.display_name}!')
        # Check if the member left the channel
        if before.channel and not after.channel:
            # send goodbye message at main channel
            await main_channel.send(f"Goodbye {member.display_name}!")
        
    @client.event
    async def on_message(message):
        # Avoid bot keep talking to himself
        if message.author == client.user:
            return
        
        # Cast everything to string
        username = str(message.author)
        user_message = str(message.content)
        channel = str(message.channel)
        print(f'{username} said: "{user_message}" in ({channel})')
        
        # Bot respond to people in the server only if they mentioned the bot
        if client.user.mentioned_in(message):
            await response_message(message, user_message, is_private=False)
    
    client.run(TOKEN)



    