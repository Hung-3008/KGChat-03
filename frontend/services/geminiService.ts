import { Message, Sender } from '../types';

// TODO: Integrate actual Google Gemini API here using @google/genai
// import { GoogleGenAI } from "@google/genai";

const MOCK_RESPONSES = [
  "I understand. Based on your description of the headache and dizziness, how long have these symptoms persisted specifically?",
  "Have you taken any medication for the pain so far? If so, did it provide any relief?",
  "I'm noting the sensitivity to light. This, combined with the headache, could suggest a migraine, but we need to rule out other causes. Have you had any vision changes?",
  "Thank you for that information. I'm generating a preliminary health summary on the right panel for your review.",
];

export const sendMessageToGemini = async (history: Message[], userMessage: string): Promise<string> => {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 1500));

  // TODO: Replace this mock logic with actual API call
  /*
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const model = ai.getGenerativeModel({ model: "gemini-pro" });
    const chat = model.startChat({ ... });
    const result = await chat.sendMessage(userMessage);
    return result.response.text();
  */

  // Return a random response for demo purposes
  const randomIndex = Math.floor(Math.random() * MOCK_RESPONSES.length);
  return MOCK_RESPONSES[randomIndex];
};
