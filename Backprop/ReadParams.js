'use strict';
let rls = require('readline-sync');

module.exports = function() {
   let paramLine, params;
   
   while (true) {
      paramLine = rls.question("Enter max character and number of characters: ")
       .split(' ');
      if (paramLine.length !== 2)
         console.log("Must have two entries");
      else {
         params = {
            maxChar: paramLine[0].toUpperCase().charAt(0),
            length: parseInt(paramLine[1])
         };
         
         if (params.maxChar < "A" || params.maxChar > "F")
            console.log("Max char must be between A and F");
         else if (!params.length || params.length > 10)
            console.log("Number of chars must be between 1 and 10");
         else {
            params.randRange
             = params.maxChar.charCodeAt(0) - 'A'.charCodeAt(0) + 1;
            return params;
         }
      }
   }
};