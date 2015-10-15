#include<string.h>
#include<stdio.h>
#include<stdlib.h>

int main (int argc, char const *argv[] )
{
   char c;
   FILE *fp1, *fp2;
  // opening the read file
   fp1 = fopen("test1.html", "r");
   if (fp1 == NULL) {
      puts("cannot open this file");
      exit(1);
   }
  // opening a new write text file index.html
   fp2 = fopen("index.html", "w");
   if (fp2 == NULL) {
      puts("Not able to open this file");
      fclose(fp1);
      exit(1);
   }
   // copying the test to index.html
   do
   {
      c = fgetc(fp1);
      if( c != EOF)
	 fputc(c, fp2);
   } while (c != EOF);
   // closing all files 
   fcloseall();

   return(0);
}
