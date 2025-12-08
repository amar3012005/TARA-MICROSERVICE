from sarvamai import SarvamAI
from sarvamai.play import save

client = SarvamAI(api_subscription_key="YOUR_API_SUBSCRIPTION_KEY")
# Convert text to speech
audio = client.text_to_speech.convert(
  target_language_code="en-IN",
  text="Welcome to Sarvam AI!",
  model="bulbul:v2",
  speaker="anushka"
)
save(audio, "output1.wav")
