"""
Command Line Interface for Julie.
"""

from julie.core.agent import Agent, ConversationHandler


def run_cli():
    """Run Julie in CLI mode with microphone input."""
    print("=" * 50)
    print("  JULIE - Voice Assistant")
    print("=" * 50)
    print()
    
    print("Initializing...")
    
    try:
        agent = Agent()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print("Ready!\n")
    
    # Create conversation handler with CLI callbacks
    handler = ConversationHandler(
        agent=agent,
        on_listening=lambda: print("\n" + "-" * 40),
        on_user_text=lambda text: print(f"You: {text}"),
        on_agent_text=lambda text: print(f"Julie: {text}"),
        on_error=lambda e: print(f"[{e}]"),
    )
    
    try:
        handler.start()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    run_cli()
