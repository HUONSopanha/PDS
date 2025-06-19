import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe } from '@nestjs/common';

async function bootstrap() {
  try {
    const app = await NestFactory.create(AppModule);
    app.enableCors();
    app.useGlobalPipes(new ValidationPipe({ whitelist: true, forbidNonWhitelisted: true }));
    // Optional: set a global prefix for all routes
    // app.setGlobalPrefix('api');
    const port = process.env.PORT ? Number(process.env.PORT) : 3000;
    await app.listen(port);
    console.log(`🚀 Server ready at: http://localhost:${port}/`);
  } catch (err) {
    console.error('Failed to start NestJS application:', err);
    process.exit(1);
  }
}
bootstrap();
