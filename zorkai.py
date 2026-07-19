from pyexpat.errors import messages
from xmlrpc import client

import pexpect
import sys
import os

import torch
from transformers import pipeline

# Define your Zork executable command (e.g., 'frotz ZORK1.DAT')
EXEC_DIR = "./devshane-zork"
os.chdir(EXEC_DIR)

ZORK_EXEC = "./zork" 

WALKTHROUGH_TXT=open("../walkthrough.txt", "r").read()

print("Loading LLM model... (this may take a while)")

# Initialize the pipeline with the official NousResearch repository
pipe = pipeline(
    "text-generation", 
    model="NousResearch/Hermes-2-Pro-Mistral-7B",
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

def build_zork_prompt(game_context: str, goal: str, mostrecent: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert, speedrun-level player of the 1980 text adventure Zork I. "
                "Your goal is to explore the Great Underground Empire, gather treasures, and survive. "
                "You must strictly obey the rules of the Infocom text parser:\n\n"
                "1. MOVEMENT: Move using ONLY cardinal directions: 'n' (north), 's', 'e', 'w', 'nw', 'ne', 'sw', 'se', 'u' (up), 'd' (down). Do not say 'move nw' to go northwest; just say 'nw'.\n"
                "2. VOCABULARY: Keep commands to 1-3 words. Use standard verbs: take, drop, open, close, read, examine, unlock, attack, turn on, turn off, eat, drink.\n"
                "3. SYNTAX: Use formats like 'take lantern', 'open mailbox', 'read leaflet', or 'attack troll with sword'.\n"
                "4. GROUNDING: Do NOT hallucinate items. Only interact with objects explicitly mentioned in the recent room description or your inventory.\n"
                "5. TROUBLESHOOTING: If you enter a dark room, immediately 'turn on lantern'. If you are stuck or confused, output 'look' to refresh the room description, or 'inventory' to check your items.\n\n"
                "6. GAME KNOWLEDGE (ZORK I SPECIFIC):\n"
                "   - The game begins near a white house. To enter, you must find the 'small window which is slightly ajar' (behind the house) and 'open window', then 'enter house'.\n"
                "   - The 'battery-powered brass lantern' is in the Living Room (inside the house). You MUST 'take lamp' and 'turn on lamp' before going into the dark cellar.\n"
                "   - The 'trap door' in the Living Room is hidden. You must 'move rug' to find it, then 'open trap door' to reach the cellar.\n"
                "   - Combat requires specifying the weapon. Example: 'kill troll with sword'.\n"
                "   - Complex mechanisms require specific tools. Example: 'turn bolt with wrench' (at the Dam) or 'push yellow button'.\n\n"
                "\n\n6. ZORK I CHEAT SHEET WALKTHROUGH: As follows is a full walkthrough of the game, as an example (not the exact version of the game you're playing):\n"
                f"{WALKTHROUGH_TXT}\n\n"
                "\n\n[END WALKTHROUGH]\n\n"
                "\n\nOutput ONLY the exact command text to send to the game. No conversational filler, no explanations, no quotation marks."
            )
        },
        {
            "role": "user",
            "content": f"Recent Game History:\n{game_context}\n\nThe goal is: {goal}\n\nThe most recent command was \"{mostrecent}\". What is your next command?"
        }
    ]

def get_llm_move(game_context: str, instr: str, mostrecent: str) -> str:
    """Passes the game context to the LLM and returns a single command."""
    messages = build_zork_prompt(game_context, instr, mostrecent)
    # print(messages)

    # Run generation
    outputs = pipe(messages, max_new_tokens=32)
    # print(outputs[0]["generated_text"][-1]["content"])
    
    # response = client.models.generate_content(
        # model='models/gemini-2-flash-lite',
        # contents=prompt
    # )
    
    # return response.text.strip()
    return outputs[0]["generated_text"][-1]["content"]


def main():
    # Spawn the interactive Zork process
    print("Starting Zork... (Type '/ai' to let AI take the next turn, or 'quit' to exit)")
    
    try:
        # encoding='utf-8' ensures we read strings, not bytes
        game = pexpect.spawn(ZORK_EXEC, encoding='utf-8', timeout=2)
    except pexpect.ExceptionPexpect as e:
        print(f"Failed to start Zork: {e}")
        sys.exit(1)

    game_history = ""

    while True:
        try:
            # Zork typically uses '>' as its input prompt
            game.expect(r'>')
            
            # Capture the output leading up to the prompt
            current_output = game.before + '>'
            print(current_output, end='')
            game_history += current_output
            
            # Keep history manageable to stay within context windows
            if len(game_history) > 3000:
                game_history = game_history[-3000:]

        except pexpect.TIMEOUT:
            pass
        except pexpect.EOF:
            print("\nGame terminated.")
            break

        # Await human input
        user_input = input(" ")

        # Check for our custom trigger
        if user_input.lower().startswith( '/ai'):
            print("\n[🧠 AI is analyzing the room...]")
            notdone=True
            ai_command = None
            while notdone:
                try:
                    ai_command = get_llm_move(game_history, user_input[4:], mostrecent=ai_command or user_input)
                    print(f"[🤖 AI executes]: {ai_command}\n")
                    if "goal acheived" in ai_command.lower():
                        print("Goal acheived.")
                        notdone = False
                        break
                    # Send the AI's command to the game
                    game.sendline(ai_command)
                    game_history += f" {ai_command}\n"

                    # Zork typically uses '>' as its input prompt
                    game.expect(r'>')
            
                    # Capture the output leading up to the prompt
                    current_output = game.before + '>'
                    print(current_output, end='')
                    game_history += current_output
            
                    # Keep history manageable to stay within context windows
                    if len(game_history) > 3000:
                        game_history = game_history[-3000:]

                except Exception as e:
                    print(f"[!] Error contacting Gemini: {e}")
                
                
        elif user_input.lower() in ['quit', 'exit']:
            game.sendline('quit')
            break
            
        else:
            # Send normal human input to the game
            game.sendline(user_input)
            game_history += f" {user_input}\n"

if __name__ == "__main__":
    main()
