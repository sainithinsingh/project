#include<stdio.h>
#include<string.h>
 
int main() {
   FILE *fp1, *fp2, *fp3;
   int ch1, ch2;
   char line1[25];
   char line2[25];
   fp1 = fopen("/tmp/testIP", "r");
 
   if (fp1 == NULL) {
      printf("Cannot open testIP for reading ");
//      return 1;
   }
   while (fscanf(fp1,"%s",line1) != EOF)
   {
     fp2 = fopen("/tmp/puppet_testIP", "r");
     if (fp2 == NULL) {
         printf("Cannot open puppet_testIP for reading ");
//        return 1;
     }
     fp3 = fopen("/tmp/temp", "w");
     if (fp3 == NULL) {
         printf("Cannot open temp for writing ");
//      return 1;
     }
     while (fscanf(fp2,"%s",line2) != EOF)
     {	
       if(strcmp (line1, line2)!=0)
       {
         fprintf(fp3,"%s\n",line2);
       }
       else 
       printf("success\n");
     }  
      fclose(fp3);
      fclose(fp2);
      system("cp /tmp/temp /tmp/puppet_testIP");
   }  
   fclose(fp1);
//   return 0;
}
