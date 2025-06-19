import { Injectable } from '@nestjs/common';
import axios from 'axios';

@Injectable()
export class OpenAIService {
  // Summarize skills from free-form input with Gemini API
  async summarizeSkillsFromInput(userInput: string): Promise<string> {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return Promise.reject({
        message: 'GEMINI_API_KEY not set in environment variables.',
        status: 403,
      });
    }

    const prompt = `
Extract the professional and technical skills mentioned in the text below.
Return only a JSON array of skill names. No explanation or extra text.

Text: "${userInput}"
`;

    try {
      const response = await axios.post(
        'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent',
        {
          contents: [
            {
              parts: [
                {
                  text: prompt,
                },
              ],
            },
          ],
        },
        {
          headers: {
            'Content-Type': 'application/json',
            'x-goog-api-key': apiKey,
          },
        },
      );

      const text = response.data?.candidates?.[0]?.content?.parts?.[0]?.text;
      if (!text) return '';

      const match = text.match(/\[.*?\]/s);
      if (!match) return '';

      const skills: string[] = JSON.parse(match[0]);
      return skills.join(', ');
    } catch (error: any) {
      if (axios.isAxiosError(error) && error.response) {
        console.error('Gemini API error:', error.response.data);
      } else {
        console.error('Error calling Gemini API:', error);
      }
      return '';
    }
  }
}
