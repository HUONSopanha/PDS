import { Entity, PrimaryGeneratedColumn, Column } from 'typeorm';

@Entity('titles')  // table name: titles
export class Title {
  @PrimaryGeneratedColumn()
  id: number;

  @Column({ nullable: false })
  Skills: string;

  @Column({ nullable: false })
  Title: string;

  // Optionally add timestamps:
  // @CreateDateColumn()
  // createdAt: Date;

  // @UpdateDateColumn()
  // updatedAt: Date;
}
