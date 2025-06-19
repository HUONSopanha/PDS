import { Controller, Post, Body } from '@nestjs/common';
import { OpenAIService } from './openai.service'; // adjust import path as needed

@Controller('skills')
export class SkillsController {
  constructor(private readonly geminiService: OpenAIService) {}

  @Post('extract')
  async extractSkills(@Body('userInput') userInput: string) {
    try {
      const skills = await this.geminiService.summarizeSkillsFromInput(userInput);
      return { skills };
    } catch (error: any) {
      if (error.status === 429) {
        return {
          statusCode: 429,
          message: error.message,
        };
      }
      throw error;
    }
  }
}
