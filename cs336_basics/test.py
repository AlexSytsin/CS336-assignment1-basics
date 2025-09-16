# from tokenizer import pretokenize
# from sortedcontainers import SortedDict
import einops
import torch
import numpy
# string = "okfこん"

# byte = string.encode("utf-8")
# z = bytes([1]) + bytes([2])
# print(type(z), z)
# print(bytes([])
import cProfile
import pstats
import pickle
import regex as re
from tokenizer import Tokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

text = """
Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed!  
He said, “Wow, that is a really amazing vase! Can I buy it?” 
The shopkeeper smiled and said, “Of course you can. You can take it home and show all your friends how amazing it is!”
So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was. 
And that's how Ben found an amazing vase in the store!
<|endoftext|>
Once upon a time, there was a reliable otter named Ollie. He lived in a river with his family. They all loved to play and swim together.
One day, Ollie's mom said, "Ollie, hurry and get some fish for dinner!" Ollie swam fast to catch fish. He saw his friend, the duck. "Hi, Ollie!" said the duck. "Hi, duck!" said Ollie. "I need to hurry and catch fish for my family."
While Ollie was catching fish, he found a big shiny stone. He thought, "This is not a fish, but it is so pretty!" Ollie took the shiny stone home to show his family. They all looked at the shiny stone and smiled. The shiny stone made everyone happy, and they forgot about the fish for dinner.
<|endoftext|>
One day, a little boy named Tim went to the park. He saw a big tiger. The tiger was not mean, but very easy to play with. Tim and the tiger played all day. They had lots of fun.
Then, something unexpected happened. The tiger started to shake. Tim was scared. He did not know what was going on. But then, the tiger turned into a nice dog. Tim was very surprised.
Tim and the dog played together now. They were very happy. The dog was easy to play with too. At the end of the day, Tim went home with his new friend.
<|endoftext|>

Once upon a time there was a friendly little boy called Bob. Bob loved to pick flowers and look for birds. One day he decided to go outside with his friends to pick some more flowers. 
He suddenly noticed something weird on the ground. It was a big, green thumb! It was so big, Bob had never seen one before. Bob curiously leaned in to take a better look. He told his friends: "look everyone, I picked up this big thumb! What do we do with it?"
His friends were very excited. They told him to pick it up and take it home to show his family. So Bob carefully picked up the friendly thumb and carried it back home. When he arrived, Bob happily showed the thumb to his family. His dad was amazed and hugged Bob to show his appreciation.
From that day on Bob always kept the big, friendly thumb with him as a reminder that special things can be found anywhere.
<|endoftext|>
Once upon a time, in a small house, there lived a little girl named Lucy. Lucy loved the color orange. She had an orange dress, an orange ball, and even an orange cat. One day, Lucy met a new friend. This friend was not like other friends. It was a spirit. The spirit was very nice and liked to play with Lucy.
One day, Lucy and the spirit were playing with her orange ball. They were having so much fun. Then, Lucy's mom called her for dinner. Lucy said to the spirit, "I have to go eat now. Will you play with me later?" The spirit nodded and smiled.
At dinner, Lucy told her mom about the spirit. But her mom did not believe her. She said, "Spirits are not real, Lucy. You have a big imagination." Lucy felt sad that her mom did not believe her. After dinner, she went back to play with the spirit. They played with the orange ball and had lots of fun. Lucy knew that even if others ignore her friend, the spirit was real and they could play together.
<|endoftext|>
One day, a boy named Tim went to the park to play. He saw his friend, Sam, playing with a toy car. Tim wanted to join and play with Sam. They both played with the toy car, and it went fast. The car had a battery inside that made it go.
Tim said, "Sam, the battery is tight in the car. It will not fall out." Sam smiled and they kept playing. They raced the car around the park, laughing and having fun.
But then, something unexpected happened. A big dog came and took the toy car in its mouth! Tim and Sam were scared, but the dog just wanted to play too. They all played together, and the dog was very gentle with the car. In the end, Tim, Sam, and the dog became good friends.
<|endoftext|>
Once upon a time, there was a sailor named Tom. Tom had a big boat. He liked to sail on the sea. One day, Tom saw a little fish. The fish was sad. It was lost and wanted to go home.
Tom said, "I can help you, little fish. You can fit in my boat, and I will take you home." The little fish was happy. It jumped into Tom's boat. They sailed together on the sea.
The sea was safe and calm. Tom and the little fish talked and laughed. They became good friends. At last, they found the fish's home. The little fish said, "Thank you, Tom, for helping me." Tom smiled and waved goodbye. He sailed away, knowing he had a new friend.
<|endoftext|>
Once upon a time, there was a smooth vase. It was very pretty. The vase lived in a small house with a girl named Lily and her mom.
One day, Lily and her mom went outside to play. Before they left, her mom said, "Lily, please close the door." Lily closed the door and they played all day.
When they came back, they saw the smooth vase on the floor. It was broken! Lily and her mom were very sad. They could not fix the pretty vase. The end.
<|endoftext|>


Sara and Ben wanted to decorate a bowl for their mom. They found a big bowl in the kitchen and some paint and brushes. They took the bowl and the paint to the backyard and put them on a table.
"Let's make the bowl pretty with colors," Sara said.
"OK, I will paint a flower," Ben said.
They started to paint the bowl with different colors. Sara painted a red heart and Ben painted a yellow flower. They were having fun.
But then, it started to rain. The rain was wet and cold. It made the paint run and drip. The bowl looked messy and ugly.
"Oh no, the rain ruined our bowl!" Sara cried.
"Mom will not like it," Ben said.
They ran inside the house with the bowl. They were sad and wet.
They showed the bowl to their mom. They said they were sorry.
But mom smiled and hugged them. She said she loved the bowl and them.
"It's a beautiful bowl," she said. "You made it with love and creativity. The rain made it special. It's like a rainbow bowl."
Sara and Ben felt happy. They gave mom a kiss and thanked her. They learned that sometimes, things can be good even when they seem bad.
<|endoftext|>
Once upon a time, in a small town, there was a deaf boy named Tom. Tom could not hear, but he was very good at seeing and feeling things. He had many friends who loved to play with him. They would play games like tag and hide-and-seek. Tom was very fast and good at finding his friends.
One day, the town bell would ring at noon. Everyone was excited to hear the bell, but Tom knew he could not hear it. He felt sad, but his friends had a plan. They decided to help Tom feel the bell ring too. They took Tom to the big bell and let him touch it while it rang. When the bell rang, Tom could feel the strong vibrations and he smiled.
The moral of the story is that even if someone is different or has a problem, like Tom being deaf, we can still find ways to help them and make them happy. We should always be kind and caring to everyone, no matter how different they are.
<|endoftext|>


Tom and Lily were twins who liked to play in the park. One day, they saw a large maze made of green bushes. They wanted to try it and find the way out.
They ran inside the maze and followed the paths. Sometimes they turned left, sometimes they turned right. They saw many flowers and birds, but they did not see the exit. They started to feel lost and scared.
They heard a loud roar behind them. It was a big lion who lived in the maze. He was hungry and angry. He wanted to catch Tom and Lily and eat them.
Tom and Lily ran away from the lion. They looked for a place to hide. They saw a small hole in the bushes. They crawled inside and hoped the lion would not find them.
The lion came closer and closer. He smelled Tom and Lily in the hole. He reached his paw inside and tried to grab them. He pinched Lily's leg with his sharp claws. Lily cried out in pain.
Tom was brave and smart. He saw a stick on the ground. He picked it up and poked the lion's eye with it. The lion howled and backed away. He was hurt and afraid.
Tom and Lily saw their chance. They ran out of the hole and out of the maze. They found their mom and dad waiting for them. They hugged them and told them what happened. They were happy and safe. They never went back to the maze again.
<|endoftext|>

Once upon a time there was a person named Sue. She weighed a lot and her momma said that she had to start eating less and exercising more. So, Sue decided to start walking around the block every day.
The first day Sue felt like it was too hard. But after a few days, Sue found it was easy! She started to feel lighter and lighter everyday.
Soon Sue wanted to do something more fun for exercise. Sue's momma suggested that she try dancing to her favorite songs on the television. Sue thought this was a great idea.
So, Sue watched the television, picked out her favorite songs and started to dance. It was hard at first but because it was fun, she soon found it was easy. And soon, Sue was weighing barely anything at all!
<|endoftext|>
One day, a little boy named Tim went to the park with his mom. Tim loved to play on the swings and slide. His mom was always dependable and watched him closely. They had a lot of fun together.
Soon, Tim told his mom, "I need to use the toilet." His mom said, "Okay, let's go find one." They found a toilet near the park. But the toilet was old and not very clean. Tim's mom didn't like it, but Tim really needed to go.
Tim went inside the toilet and tried to use it. But, suddenly, the toilet broke and water came out. Tim got very wet and dirty. He cried and said, "Mom, I don't like this toilet!" His mom hugged him and said, "I know, Tim. We will find a better one next time." They went home with Tim still feeling sad and wet.
<|endoftext|>
One day, a little girl named Lucy went to the park with her mom. They saw a big tree with lots of flowers. The flowers had sweet smells. Lucy liked the smells. She wanted to count the flowers on the tree.
"Mom, can I count the flowers?" Lucy asked. Her mom said, "Yes, you can count the flowers. Be careful not to touch them."
Lucy started to count. "One, two, three," she said. But then, she saw something strange. There was some powder on the flowers. She touched the powder and it was soft. She showed her mom.
"Look, mom! What is this powder?" Lucy asked. Her mom smiled and said, "That is flower powder. It helps make more flowers."
Lucy was happy to learn something new. She finished counting the flowers and they went home. Now, Lucy knew that the sweet-smelling flowers had special powder to help them grow.
<|endoftext|>
Once upon a time, there was a hunter. He lived in a small house near the woods. He had a dog named Spot. Spot was a fierce dog. They were best friends.
One day, the hunter and Spot went into the woods to find food. They saw a big bird in a tree. The hunter got ready to shoot. Spot barked loudly. The bird flew away. They did not catch the bird, but they were not sad.
The hunter and Spot played in the woods. They ran and jumped. They had lots of fun. Then, they went back to their small house. They were happy to be together. And they lived happily ever after.
<|endoftext|>
Once, there was a young boy named Tim. Tim had a toy spear. He liked to play with his spear all day. One day, Tim went to the park to play with his friends.
At the park, Tim saw a big tree. Tim dared his friend, Sam, to climb the tree. Sam said, "I dare you to climb the tree first!" So, Tim started to climb the tree with his spear.
When Tim got to the top of the tree, he saw a big bird. The bird looked at Tim and said, "Hello! I am a talking bird!" Tim was very surprised. He did not know birds could talk. The bird became Tim's new friend, and they played together in the park all day.
<|endoftext|>
One day, a girl named Lily found a broken toy car. She wanted to repair it and make it pretty. So, she went to her dad for help. 
"Daddy, can you help me fix this car and make it pretty?" Lily asked. Her dad said, "Of course, let's use some paint to make it pretty too!" They both worked together to repair the toy car.
After they fixed the car, they decided to paint it. They painted it with many colors. When they were done, the car looked very pretty. But then, something unexpected happened. The car started to move on its own! Lily and her dad were very surprised. They had so much fun playing with the magical toy car.
<|endoftext|>

Once upon a time there was a little girl named Kate. She had a wonderful garden in her backyard. She loved to play in it.
One day, something strange happened in the garden. All the flowers in the garden were starting to turn brown, and it made Kate very worried. 
Kate went to her mom and said, "Mom, my garden looks terrible! What should I do?"
Her mom said, "Don't worry, honey. We'll figure it out together."
They went to the garden and worked to fix it. They watered the flowers and got rid of the bugs. Soon, the garden was beautiful again.
Kate was very happy and she smiled. She thanked her mom for helping her. She knew she would always have someone to rely on when she was worried.
<|endoftext|>


Sam and Ben are friends. They like to play in the park. They see a big dog with a guard. The guard is bald. He has no hair on his head.
"Look at the dog!" Sam says. "He is so big and furry!"
"Can we pet him?" Ben asks. "He looks nice."
They walk to the guard and the dog. The guard smiles at them.
"Hello, boys," he says. "This is Rex. He is a good dog. He likes children. Do you want to pet him?"
"Yes, please!" Sam and Ben say.
They reach out their hands. Rex wags his tail. He licks their faces. Sam and Ben giggle.
They pet Rex for a while. They feel his soft fur. They gaze at his brown eyes. They see his sharp teeth.
"Wow, he is so cool!" Sam says.
"Thank you for letting us pet him," Ben says.
"You're welcome," the guard says. "Rex likes you. You are very polite."
Sam and Ben say goodbye to the guard and Rex. They run back to the slide. They are happy. They have a new friend.
<|endoftext|>
One day, a little girl named Amy was eager to wear her new dress. It was a pretty dress with lots of colors. She wanted to show it to her friend, Ben.
Amy went to Ben's house and said, "Look at my new dress, Ben! Isn't it pretty?" Ben looked at the dress and said, "Yes, it is very pretty, Amy. I like the colors."
They played outside all day, running and jumping. But then, Amy fell down and got her dress dirty. She started to cry. Ben said, "I am sorry, Amy. I did not want you to get your dress dirty."
Amy felt regret for playing so hard in her new dress. She said, "It's okay, Ben. Next time, I will be more careful when I wear my new dress."
From that day on, Amy learned to be more careful when wearing her pretty dresses. And she and Ben continued to play and have fun together.
<|endoftext|>
Once upon a time, there was a pale lion who lived in a tunnel. The lion was not happy because he could not roar like the other lions. Every time he tried to roar, only a small sound came out. This made the lion very sad.
One day, a little bird came to the tunnel. The bird told the lion that there was a magic fruit in the forest that could help him roar. The lion was excited and went to look for the fruit. He found it under a big tree and ate it right away.
After eating the magic fruit, the lion felt a strange feeling in his throat. He tried to roar, and to his surprise, he roared so loud that it shook the whole forest. The other lions came running to see what happened. They saw the pale lion and were amazed by his big roar. The lions became friends and lived happily together, and the pale lion was never sad again.
<|endoftext|>
Once upon a time, there was a big pumpkin. It was hot outside, and the pumpkin was sad. It wanted to be cool like the other pumpkins in the garden. The pumpkin saw a bird sitting on a tree branch and asked, "Can you teach me how to be cool like the other pumpkins?"
The bird said, "I can teach you how to make a big shade. Then you will be cool like the other pumpkins." The pumpkin was happy to learn from the bird. So, they worked together to make a big shade using leaves and sticks.
But then, a strong wind came and blew the shade away. The pumpkin was sad again. The bird told the pumpkin, "It's okay. Sometimes things don't work out the way we want them to. The important thing is to keep trying and never give up."
The pumpkin learned to be strong and keep trying, even when things were hard. And soon, the weather changed, and the pumpkin was cool and happy like the other pumpkins. The moral of the story is to never give up and keep trying, even when things are tough.
<|endoftext|>


Sara and Tom were playing in the garden. They liked to look for bugs and flowers. Sara had a mint plant in a pot. She liked to smell it and taste it. Tom wanted to try it too.
"Can I have some mint?" he asked Sara.
"Sure, here you go," Sara said. She gave him a leaf. Tom put it in his mouth. He made a face. It was very strong and spicy.
"Yuck, I don't like it!" he said. He spat it out. Sara laughed.
"It's good for you," she said. "It makes your breath fresh."
Just then, they heard a loud noise. It was a dog barking. They looked up and saw a big black dog running towards them. It had sharp teeth and a long tongue. It looked angry.
"Help, help!" Sara and Tom shouted. They were frightened. They dropped the mint and ran to the house. The dog chased them.
"Go away, go away!" they cried.
Their mom heard them and came out. She saw the dog and yelled at it.
"Bad dog, bad dog! Leave my kids alone!" she said. She picked up a stick and waved it at the dog. The dog stopped and ran away.
Sara and Tom hugged their mom. They were safe.
"Thank you, mom," they said.
"You're welcome, my loves," she said. "Are you okay?"
"Yes, we're okay," they said.
They told her about the mint and the dog. She smiled and kissed them.
"I'm glad you're okay," she said. "But next time, be careful. Don't go near strange dogs. And don't eat too much mint. It might give you a tummy ache."
<|endoftext|>


Tom and Lily are friends. They like to play pretend. Today, they pretend to be a post office. Tom is the postman. Lily is the customer.
Tom knocks on the door of Lily's house. He has a big bag of letters and packages. He says, "Hello, I am the postman. I have some post for you."
Lily opens the door and smiles. She says, "Hello, postman. Thank you for bringing me the post. You are very polite."
Tom gives Lily the post. He says, "You are welcome, customer. I like to perform my job well. Do you have any post to send?"
Lily thinks for a moment. She says, "Yes, I do. I have a card for my grandma. Can you take it to her?"
Tom nods. He says, "Of course, I can. I will deliver it fast and safe. Where does your grandma live?"
Lily says, "She lives in the yellow house on the hill. You can see it from here. It has a big garden and a swing."
Tom says, "I know that house. It is very pretty. I will go there now. Goodbye, customer."
Lily says, "Goodbye, postman. Have a nice day."
Tom and Lily wave to each other. They are happy. They like to play pretend. They are good friends.
<|endoftext|>

Once upon a time there was a fish who was very happy in the water. He swam around with his fish friends and spent all day exploring the deep blue ocean.
One day, the fish did something very foolish. He swam too close to the surface and ate a fisherman’s bait. As soon as he did, the fisherman pulled him out of the water. The fish was so scared and he felt so sad.
But the fisherman was very kind. He knew that the fish had made a mistake, so he put him back in the water and said “I forgive you”.
The little fish was so grateful, and he learned an important lesson – never do something foolish again! He made sure he listened to the wise fish and stayed away from the surface.
The moral of this story is to always be kind and to forgive those who make mistakes. It’s important to be patient and understanding with others, even when they do something foolish.
<|endoftext|>
Once upon a time, there was a nice girl named Lily. She loved to play with her friends outside. They liked to run, jump, and laugh in the fresh air. The air was full of oxygen, which helped them breathe and have fun.
One day, Lily and her friends found a big log. They thought it would be fun to walk on it. They took turns trying to balance on the log. Some of them wobbled and almost fell, but they all tried their best.
In the end, Lily was the best at balancing on the log. Her friends clapped and cheered for her. They all had a nice day playing outside, breathing the oxygen, and learning to balance. And they all lived happily ever after.
<|endoftext|>
One day, a little girl named Amy found a baby bird in her yard. The baby bird could not fly. Amy wanted to help the baby bird, so she picked it up and took it to her mom. "Mom, can we help this baby bird?" Amy asked. Her mom shrugged and said, "We can try, but it might be hard."
Amy and her mom took the baby bird inside and gave it some food and water. They made a warm bed for the baby bird to sleep in. The next day, Amy went to her friend Tom's house. She told Tom about the baby bird. Tom said, "I don't think you should help the baby bird. It's not your job." But Amy knew it was the right thing to do. She said, "I want to be honest and kind, so I will help the baby bird."
Amy and her mom took care of the baby bird for many days. They helped it get strong and learn to fly. One day, the baby bird was ready to go back to the wild. Amy and her mom took the baby bird outside and said goodbye. The baby bird flew away, happy and free. Amy felt good because she had been honest and kind. The moral of the story is to always be honest and kind, even when it's hard.
<|endoftext|>
Once upon a time, there was a little boy named Tim. Tim loved to play sport. His favorite sport was soccer. One day, Tim was playing soccer with his friends. They were having so much fun. But then, something terrible happened. Tim's ball had a hole in it.
"Oh no! My ball has a hole!" Tim said. His friends looked sad too. They all wanted to keep playing, but they couldn't. Tim's mom heard them and came outside. She saw the hole in the ball and said, "Don't worry, Tim. I can fix it."
Tim's mom went inside the house and came back with a patch. She put the patch on the hole and fixed the ball. Tim and his friends were so happy. They all said, "Thank you!" to Tim's mom.
Now, Tim and his friends could play soccer again. They played until the sun went down. They had a great day playing their favorite sport. And Tim learned that when something is terrible, it can be fixed with a little help from his mom.
<|endoftext|>

Once upon a time there was a boy named Thomas. He was a brave three-year-old who loved to play football.
Every Saturday, he went to his local park to play with his friends. Today, something different happened.
When they walked into the park, they heard a loud noise. Suddenly, a fierce lion entered the park! Everyone was scared, but Thomas stood still. He saw his football lying on the ground and he got an idea.
He bravely stepped forward and threw the football at the fierce lion. The lion was so surprised it ran away! Everyone celebrated and Thomas became a hero!
<|endoftext|>
"""

tokenizer = Tokenizer.from_files("/Users/alex/CS336/CS336-assignment1-basics/tokenizers/owt_train.pkl")

    # print(len)
# text= "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
# special_tokens=["<|endoftext|><|endoftext|>", "<|endoftext|>"]
# escaped_tokens = [re.escape(token) for token in special_tokens]
# delimiter = "|".join(escaped_tokens)
# new_pattern = f"{delimiter}"
# print(new_pattern)
# matches = re.findall(new_pattern, text)
# print(matches)
# special_tokens = set([" "])
# escaped_tokens = [re.escape(token) for token in sorted(special_tokens, reverse=True, key=len)]
# delimiter = "|".join(escaped_tokens)
# print(delimiter)
# print(f"({delimiter})")
# chunks_and_special_tokens = re.split(f"({delimiter})", text)
# print(len(chunks_and_special_tokens))

import einops #version 0.8.1
import torch 

a = torch.ones(3,2)
b = torch.ones(2)
c = einops.einsum(a, b, "out_dim in_dim, in_dim -> out_dim")
typo_c = einops.einsum(a, b, "out_dim in_dim, in_dimm -> out_dim in_dim")
print(a)
print(b)
print(c)
print(typo_c)
assert (c == typo_c).all()