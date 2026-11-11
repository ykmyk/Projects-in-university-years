using FocusTube.Services;

var builder = WebApplication.CreateBuilder(args);

// Fetch YouTube's API Key from the configuration file (appsettings.json).
// If it is missing, it throws an immediate exception.
// This prevents the app from booting in a broken state.
var apiKey = builder.Configuration["YouTubeSettings:ApiKey"] 
             ?? throw new Exception("YouTube API Key is missing!");

// Register the YouTube service as a Singleton, meaning that only one instance of this service will 
// be created and shared across the entire application and save the memory.
builder.Services.AddSingleton<IYouTubeIntegrationService>(new YouTubeIntegrationService(apiKey));

// Register the core domain logic service, WatchGuardService, as a Singleton as well.
// This holds the in-memory `_history` list, thus important to be a Singleton.
// If not, the user's watch history would reset to zero every time the user refreshed the page.
builder.Services.AddSingleton<WatchGuardService>(); 

// Add support for API Controllers.
// This allows to rerout requests like /api/videi to the right C# classes
builder.Services.AddControllers();

// Add Swagger tools for auto-generating API documentation and testing interfaces.
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Enable the Swagger UI in Development mode. 
// This creates a nice webpage at /swagger 
// allowing us(developers) to test API endpoints without writing HTML.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

// Redirect all standard HTTP traffic to secure HTTPS.
app.UseHttpsRedirection();

// Serve the 'index.html' file automatically(as a default) 
// when a user visits the root URL (http://localhost:5239/).
app.UseDefaultFiles();

// Allow the server to deliver static assets like our HTML pages, 
//CSS files, and JavaScript scripts from the wwwroot folder.
app.UseStaticFiles();

// Map incoming URL requests (e.g., /api/settings) to the corresponding Controller classes.
app.MapControllers(); 

// Start the web server and begin listening for incoming requests.
app.Run();