


// Gemma chatbot

using AIGemmaAssistantApp.Models;
using AIGemmaAssistantApp.Services;


//GemmaChatbotService gemmaChatbotService = new GemmaChatbotService();
//gemmaChatbotService.StartChat();

string modelPath = Path.Combine(
    AppContext.BaseDirectory,
    "AIModels",
    "Gemma2-2B-INT4");

using var chatService = new GemmaChatService(modelPath);

var history = new List<ChatMessage>();

Console.WriteLine("Gemma is ready!");
Console.WriteLine("Type 'exit' to quit.");
Console.WriteLine("Type 'clear' to clear history.");
Console.WriteLine("Type 'help' to see available commands.");
Console.WriteLine();

while (true)
{
    Console.Write("\nYou: ");

    string? userInput = Console.ReadLine();

    if (userInput == null)
        break;

    userInput = userInput.Trim();

    if (string.IsNullOrWhiteSpace(userInput))
        continue;

    if (userInput.Equals("exit", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine("Assistant Closed.");
        break;
    }
        

    if (userInput.Equals("clear", StringComparison.OrdinalIgnoreCase))
    {
        history.Clear();
        Console.WriteLine("Conversation history cleared.");
        continue;
    }

    if (userInput.Equals("help", StringComparison.OrdinalIgnoreCase))
    {
        Console.WriteLine();
        Console.WriteLine("========== Gemma AI Assistant ==========");
        Console.WriteLine("Ask anything       - Send a question to Gemma");
        Console.WriteLine("clear              - Clear conversation history");
        Console.WriteLine("help               - Show available commands");
        Console.WriteLine("exit               - Close the chatbot");
        Console.WriteLine("========================================");
        Console.WriteLine();

        continue;
    }

    var messages = new List<ChatMessage>(history)
    {
        new("user", userInput)
    };

    try
    {
        var answer = chatService.GetResponse(messages);
        Console.WriteLine($"Gemma: {answer}");
        history.Add(new("user", userInput));
        history.Add(new("assistant", answer));

        while (history.Count > 8)
            history.RemoveRange(0, 2);

      
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: {ex.Message}");
    }

   
}

Console.ReadKey();