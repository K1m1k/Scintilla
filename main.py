from consciousness.consciousness import ConsciousnessSystem

def main():
    """
    Main entry point for the project.
    """
    consciousness_system = ConsciousnessSystem()
    raw_inputs = [
        "I am feeling happy today.",
        "The weather is sunny and warm.",
        "I had a great conversation with my friend.",
        "I am learning new things every day."
    ]
    consciousness_system.simulate_consciousness(raw_inputs)
    output = consciousness_system.produce_output()
    print(output)

if __name__ == "__main__":
    main()
